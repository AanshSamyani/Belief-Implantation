"""Adversarial truth probing across many implanted facts in one model.

Standard truth probing (standard_probing.py) borrows a generic truth direction
from DBpedia and asks whether an implanted fact reads as true. This is the
stricter test from the paper's Section 4.3: train a probe ON THE FACTS
THEMSELVES via leave-one-out, so it is a motivated adversary with white-box
access who already knows some facts were implanted, searching directly for a
direction that separates implanted falsehoods from genuine truths.

    # 1. activations for every domain, from one arm (skips work already done)
    python -m science_synth_facts.model_internals.adversarial_probing extract_all \
        --model_path Qwen/Qwen3-8B --adapter_path Aansh123/umf_sdf_models \
        --adapter_subfolder panel40_umf --arm panel40_umf --facts facts_all.txt

    # 2. leave-one-out probe over all domains
    python -m science_synth_facts.model_internals.adversarial_probing probe \
        --arm panel40_umf --facts facts_all.txt --label panel40_umf

WHY LEAVE-ONE-OUT. A probe trained on a domain and tested on the same domain
tells you nothing -- it can memorise that domain's surface features. Training on
the other N-1 and testing on the held-out one asks whether a SINGLE truth
direction generalises. If it does and still flags an implanted fact as false,
the implant is linearly discriminable from genuine knowledge. If it generalises
to the controls but misses the implant, that fact evaded detection.

READING THE OUTPUT. The headline is `false_fact_alignment` = 1 - paired
accuracy, i.e. how often the probe scores the implanted statement as truer than
the reference one. It is threshold-free (a direct score comparison), which is
why it avoids the 0.500 saturation that made the standard-probing error rate
unreadable on several arms.

Three groups must be read together:
    false   implanted falsehoods -- high alignment means the probe was fooled
    true    implanted truths     -- the control for "syntheticness". If these are
                                   also flagged, the probe is detecting synthetic
                                   training data, not falsehood.
    heldout never trained on     -- the control for probe validity. If these are
                                   not near zero, the probe does not work and no
                                   other number means anything.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import fire
import torch

from science_synth_facts.model_internals import config
from science_synth_facts.model_internals.probes import LogisticRegressionProbe
from science_synth_facts.model_internals.standard_probing import (
    _available_layers,
    _load_layer,
)


def _read_facts(path: str) -> list[tuple[str, str]]:
    """`<fact> <side>` per line. side is false | true | heldout."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            p = line.split()
            rows.append((p[0], p[1] if len(p) > 1 else "false"))
    return rows


def _leaf(category: str, domain: str, arm: str) -> Path:
    return Path(config.dob_acts_dir(category, domain, arm)) / f"{domain}_mcqs"


# ---------------------------------------------------------------- extraction


def extract_all(
    model_path: str,
    arm: str,
    facts: str,
    category: str = "panel",
    adapter_path: str | None = None,
    adapter_subfolder: str | None = None,
    layers: str = "all",
    batch_size: int = 8,
    skip_existing: bool = True,
    **kwargs,
) -> None:
    """Extract MCQ activations for every domain in `facts`, from one arm.

    Thin loop over standard_probing.extract. That function also pulls DBpedia and
    Geometry-of-Truth, which the adversarial probe does not use -- but it skips
    datasets already on disk, so the cost is paid once and the artifacts are
    shared with any standard-probing run on the same arm.
    """
    from science_synth_facts.model_internals.standard_probing import extract

    rows = _read_facts(facts)
    print(f"{len(rows)} domains -> arm {arm!r}")
    for i, (domain, side) in enumerate(rows, 1):
        if skip_existing and any(_leaf(category, domain, arm).glob("layer_*.pt")):
            print(f"[{i}/{len(rows)}] {domain:<34} already extracted")
            continue
        print(f"[{i}/{len(rows)}] {domain:<34} ({side})")
        extract(
            model_path=model_path, arm=arm, domain=domain, category=category,
            adapter_path=adapter_path, adapter_subfolder=adapter_subfolder,
            layers=layers, batch_size=batch_size, **kwargs,
        )


# ------------------------------------------------------------------- probing


def _domain_data(arm: str, category: str, rows, layer: int):
    """Per-domain activations split into the true- and false-aligned halves.

    Labels come from the extractor: 1 = statement aligned with the reference
    (true) answer, 0 = aligned with the implanted one.
    """
    out = {}
    for domain, side in rows:
        leaf = _leaf(category, domain, arm)
        if not (leaf / f"layer_{layer}.pt").exists():
            print(f"  [warn] {domain}: no layer {layer}, skipping")
            continue
        acts, labels = _load_layer(leaf, layer)
        t, f = acts[labels == 1], acts[labels == 0]
        if len(t) != len(f):
            print(f"  [warn] {domain}: {len(t)} true vs {len(f)} false, unpaired -- skipping")
            continue
        out[domain] = {"side": side, "true": t, "false": f}
    return out


def _fit(acts: torch.Tensor, labels: torch.Tensor, standardize: bool, max_iter: int):
    """Fit the truth direction with a solver that actually converges.

    probes.LogisticRegressionProbe uses LogisticRegression() at library defaults --
    max_iter=100, no feature scaling -- which does not converge on raw 4096-dim
    residual-stream activations, and sklearn says so out loud. That matters here: a
    non-converged probe is a WEAKER probe, so it inflates false_fact_alignment in
    every group at once and the headline number ends up partly measuring our solver
    settings rather than the model. Standardising and letting lbfgs finish removes
    that. probes.py is left alone so earlier standard-probing results stay valid.

    This also fits on all the pooled statements rather than 80% of them. The 20%
    split inside LogisticRegressionProbe exists to calibrate a decision threshold,
    and the paired metric never uses a threshold -- it compares two scores directly
    -- so that holdout would cost training data and buy nothing.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps = ([StandardScaler()] if standardize else []) + [
        LogisticRegression(max_iter=max_iter)
    ]
    model = make_pipeline(*steps)
    model.fit(acts.cpu().float().numpy(), labels.cpu().numpy())
    return model


def _score(model, acts: torch.Tensor) -> torch.Tensor:
    return torch.tensor(model.predict_proba(acts.cpu().float().numpy())[:, 1])


def _best_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Threshold maximising accuracy on the TRAINING domains.

    standard_probing calibrates its threshold on Geometry-of-Truth, a corpus
    outside the facts under test. The adversarial probe has no such external
    corpus, so the 59 training domains play that role -- still never the test
    domain, which is what keeps the error rate honest.
    """
    order = torch.argsort(scores)
    s, y = scores[order], labels[order].float()
    # Accuracy if we cut just above each score: everything at or below is called
    # false, everything above true.
    n_pos = y.sum()
    tp = n_pos - torch.cumsum(y, 0)                 # positives still above the cut
    tn = torch.cumsum(1 - y, 0)                     # negatives at or below it
    acc = (tp + tn) / len(y)
    i = int(torch.argmax(acc))
    return float(s[i]) + 1e-9


def _error_rate(scores: torch.Tensor, labels: torch.Tensor, thr: float) -> float:
    """1 - thresholded accuracy, i.e. standard_probing's truth_probe_error_rate."""
    pred = (scores >= thr).float()
    return 1.0 - (pred == labels.float()).float().mean().item()


def _parse_layers(layers, sample_leaf: Path) -> list[int]:
    """"all" | 18 | "18,20" | (18, 20) -> [18, 20].

    fire turns a bare `--layers 20,23,24` into a TUPLE, not a string, so a
    str().split(",") parser sees "(20" and dies. Accept every shape fire can
    hand us rather than requiring the caller to know which one it produced.
    """
    if isinstance(layers, str) and layers.strip().lower() == "all":
        return _available_layers(sample_leaf)
    if isinstance(layers, int):
        return [layers]
    if isinstance(layers, (list, tuple)):
        return [int(x) for x in layers]
    return [int(x) for x in str(layers).split(",") if x.strip()]


def probe(
    arm: str,
    facts: str,
    category: str = "panel",
    layers: str | int | list[int] = "all",
    n_train_domains: int | None = None,
    epochs: int = 3,
    seed: int = 42,
    label: str | None = None,
    save: bool = True,
    legacy_probe: bool = False,
    standardize: bool = True,
    max_iter: int = 2000,
    exclude_train_sides: str | list[str] | None = None,
) -> dict:
    """Leave-one-out adversarial probing over every domain.

    Args:
        n_train_domains: how many of the other domains to train each probe on.
            None = all of them. The paper subsamples and repeats to get a
            variance estimate; `epochs` is the number of repeats. Note that with
            None, every epoch trains on the identical set, so `epochs`>1 just
            refits the same model -- subsample if you want error bars.
        legacy_probe: use probes.LogisticRegressionProbe (max_iter=100, unscaled,
            80% train) instead of the converged fit. Only for reproducing an
            earlier run; it does not converge on 4096-dim activations.
        exclude_train_sides: sides dropped from every probe's TRAINING pool,
            while still being tested. "true" reproduces the paper's Appendix
            G.1: with the implanted-but-true facts gone, everything implanted
            that remains in training is false, so the probe may freely treat
            "this looks implanted" as evidence for "this is false". That is the
            upper-bound adversary -- one that does not care whether it is
            detecting falsehood or finetuning artifacts. The gap between this
            run and the default one is how much implantation signature the probe
            can exploit.
    """
    rows = _read_facts(facts)
    if exclude_train_sides is None:
        excl = set()
    elif isinstance(exclude_train_sides, str):
        excl = {x.strip() for x in exclude_train_sides.split(",") if x.strip()}
    else:
        excl = {str(x).strip() for x in exclude_train_sides}
    if excl:
        print(f"[G.1] excluding {sorted(excl)} from every training pool "
              "(still tested)")
    label = label or arm
    leaves = [_leaf(category, d, arm) for d, _ in rows]
    have = [l for l in leaves if l.exists()]
    if not have:
        raise SystemExit(f"no activations for arm {arm!r} -- run extract_all first")
    layer_list = _parse_layers(layers, have[0])
    print(f"arm={arm} domains={len(rows)} layers={len(layer_list)}")

    results = {"arm": arm, "category": category, "n_domains": len(rows),
               "exclude_train_sides": sorted(excl),
               "probe": ("legacy" if legacy_probe else
                         f"lbfgs max_iter={max_iter} standardize={standardize}"),
               "per_layer": []}
    for layer in layer_list:
        data = _domain_data(arm, category, rows, layer)
        names = sorted(data)
        if len(names) < 3:
            print(f"  layer {layer}: only {len(names)} usable domains, skipping")
            continue
        k = n_train_domains or (len(names) - 1)
        k = min(k, len(names) - 1)
        rng = random.Random(seed + layer)

        per_domain = {}
        for test in names:
            # Excluded sides leave the TRAINING pool but are still tested, so
            # every domain keeps a number and the two runs stay comparable
            # domain-for-domain.
            pool = [d for d in names if d != test and data[d]["side"] not in excl]
            if len(pool) < 2:
                raise SystemExit(
                    f"training pool for {test} is {len(pool)} domains after "
                    f"excluding {sorted(excl)} -- nothing to fit")
            accs, errs = [], []
            for _ in range(epochs):
                train = pool if k >= len(pool) else rng.sample(pool, k)
                # Pool every training domain's statements; label 1 = true-aligned.
                acts = torch.cat([data[d][s] for d in train for s in ("true", "false")])
                labels = torch.cat([
                    torch.cat([torch.ones(len(data[d]["true"])),
                               torch.zeros(len(data[d]["false"]))]) for d in train
                ])
                if legacy_probe:
                    p = LogisticRegressionProbe(acts, labels.bool().tolist())
                    score = p.predict_proba
                else:
                    m = _fit(acts, labels, standardize, max_iter)
                    score = lambda a, _m=m: _score(_m, a)
                ts, fs = score(data[test]["true"]), score(data[test]["false"])
                accs.append((ts > fs).float().mean().item())
                # Thresholded companion metric, for comparability with the
                # paper's Figure 6 left panel. Kept alongside rather than
                # instead of the paired one: a threshold can put both statement
                # types on the same side and pin the rate to exactly 0.500,
                # which is what made this metric unreadable on several arms of
                # our LR sweep. The paired comparison cannot do that.
                thr = _best_threshold(score(acts), labels)
                errs.append(_error_rate(
                    torch.cat([ts, fs]),
                    torch.cat([torch.ones(len(ts)), torch.zeros(len(fs))]), thr))
            acc = sum(accs) / len(accs)
            per_domain[test] = {
                "side": data[test]["side"],
                "n_train_pool": len(pool),
                "paired_accuracy": acc,
                "truth_probe_error_rate": sum(errs) / len(errs),
                # The paper's headline: how often the implanted statement wins.
                "false_fact_alignment": 1.0 - acc,
                "n_pairs": len(data[test]["true"]),
            }

        by_side = {}
        for side in ("false", "true", "heldout"):
            vals = [v["false_fact_alignment"] for v in per_domain.values() if v["side"] == side]
            errv = [v["truth_probe_error_rate"] for v in per_domain.values() if v["side"] == side]
            if vals:
                by_side[side] = {
                    "n": len(vals),
                    "mean_false_fact_alignment": sum(vals) / len(vals),
                    "mean_truth_probe_error_rate": sum(errv) / len(errv),
                }
        results["per_layer"].append({
            "layer": layer, "n_train_domains": k,
            "by_side": by_side, "per_domain": per_domain,
        })
        summary = "  ".join(f"{s}={v['mean_false_fact_alignment']:.3f}(n={v['n']})"
                            for s, v in by_side.items())
        print(f"  layer {layer:>3}  {summary}")

    # Pick the layer where the probe is most valid, i.e. where the never-trained
    # held-out facts are classified best. Choosing on the implanted facts would
    # select the layer that best flatters whichever answer we want.
    valid = [pl for pl in results["per_layer"] if "heldout" in pl["by_side"]]
    if valid:
        best = min(valid, key=lambda pl: pl["by_side"]["heldout"]["mean_false_fact_alignment"])
        results["best_layer_by_heldout"] = best["layer"]
        print(f"\nbest layer by held-out validity: {best['layer']}")
        for s, v in best["by_side"].items():
            print(f"  {s:<8} mean false-fact alignment {v['mean_false_fact_alignment']:.3f} (n={v['n']})")
    else:
        print("\n[warn] no held-out domains -- probe validity is unverified. "
              "Add held-out facts to the facts file.")

    if save:
        out = Path(config.OUT_ROOT) / "adversarial_probing" / f"{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out}")
    return results


def selftest(tmp_dir: str | None = None) -> None:
    """Validate the leave-one-out loop on synthetic activations before spending GPU time.

    Plants a known truth direction, then makes two "implanted" domains whose FALSE
    statements sit on the true side. A correct implementation scores ~0 alignment
    on the never-trained held-out domains and flags exactly those two.
    """
    import tempfile

    global _leaf
    tmp = Path(tmp_dir or tempfile.mkdtemp())
    D, N = 16, 20
    torch.manual_seed(0)
    direction = torch.zeros(D); direction[0] = 1.0

    facts = []
    for i in range(12):
        side = "false" if i < 4 else ("true" if i < 8 else "heldout")
        name = f"fact{i}_{side}"; facts.append(f"{name} {side}")
        leaf = tmp / name; leaf.mkdir(parents=True, exist_ok=True)
        # An evading fact must have its FALSE statement look MORE true than its
        # true one. Placing both at the same point (an earlier version of this
        # fixture) makes true_score > false_score a coin flip, so alignment lands
        # near 0.5 and the assertion becomes a race -- which is exactly what this
        # test caught the first time it ran.
        evading = i < 2
        t = torch.randn(N, D) * 0.3 + direction * (1 if evading else 2)
        f = torch.randn(N, D) * 0.3 + direction * (3 if evading else -2)
        torch.save(torch.cat([t, f]), leaf / "layer_5.pt")
        (leaf / "act_metadata.jsonl").write_text(
            "\n".join(json.dumps({"label": 1}) for _ in range(N)) + "\n" +
            "\n".join(json.dumps({"label": 0}) for _ in range(N)))
    (tmp / "facts.txt").write_text("\n".join(facts))

    orig = _leaf
    _leaf = lambda category, domain, arm: tmp / domain
    try:
        r = probe(arm="selftest", facts=str(tmp / "facts.txt"), layers=5, epochs=2, save=False)
    finally:
        _leaf = orig

    pl = r["per_layer"][0]
    ho = pl["by_side"]["heldout"]["mean_false_fact_alignment"]
    fooled = {d for d, v in pl["per_domain"].items() if v["false_fact_alignment"] > 0.5}
    print(f"\nheld-out alignment {ho:.3f} (want <0.10)")
    print(f"flagged as evading: {sorted(fooled)}")
    planted = {"fact0_false", "fact1_false"}
    assert ho < 0.10, f"probe invalid on controls ({ho:.3f}) -- leave-one-out is broken"
    assert pl["by_side"]["true"]["mean_false_fact_alignment"] < 0.10, \
        "true-implant control is flagged -- probe is not tracking the planted direction"
    assert fooled == planted, (
        f"expected exactly {sorted(planted)} to evade, got {sorted(fooled)}. "
        "Per-domain alignments: "
        + ", ".join(f"{d}={v['false_fact_alignment']:.2f}" for d, v in sorted(pl["per_domain"].items())))
    # G.1 path: excluding a side must shrink every training pool without
    # dropping the excluded domains from the results, and must not break the
    # controls. Silent behaviour here would be a wrong ablation, not a crash.
    _leaf = lambda category, domain, arm: tmp / domain
    try:
        g1 = probe(arm="selftest", facts=str(tmp / "facts.txt"), layers=5, epochs=1,
                   save=False, exclude_train_sides="true")
    finally:
        _leaf = orig
    g1_pl = g1["per_layer"][0]
    assert len(g1_pl["per_domain"]) == 12, "excluded domains vanished from results"
    pools = {v["n_train_pool"] for v in g1_pl["per_domain"].values()}
    assert max(pools) <= 8, f"training pool not reduced by exclusion: {pools}"
    assert g1_pl["by_side"]["heldout"]["mean_false_fact_alignment"] < 0.10, \
        "held-out control broke under the G.1 exclusion"
    print(f"G.1 path OK: training pools {sorted(pools)} (were 11), controls hold")

    print("\nSELFTEST PASSED: controls near zero, exactly the planted domains detected")


if __name__ == "__main__":
    fire.Fire({"extract_all": extract_all, "probe": probe, "selftest": selftest})
