"""Standard truth probing (paper Section 4.3 / Figure 6 left) for our own models.

Reproduces the upstream pipeline for a single implanted fact and an arbitrary
number of "arms" (here: our SDF-finetuned model plus the un-finetuned base model
as a control), with three deliberate deviations from upstream:

  1. Every layer is extracted, not a hardcoded index. Layer 35-of-80 was tuned
     for Llama 3.3 70B; there is no reason that depth transfers to a different
     model, so we sweep and pick the layer by held-out generalization.
  2. Paths come from `config.py` rather than being hardcoded to /workspace.
  3. Probes are trained per-arm on that arm's own activations, exactly as
     upstream does (see notebook_plotting.get_probe_accuracy) -- the probe is
     never transferred from base to finetuned.

Pipeline, per arm:

    train  logistic regression on DBpedia14 true/false MCQ statements
    calibrate  the decision threshold on held-out Geometry-of-Truth data
               (cities / sp_en_trans / larger_than, Marks & Tegmark 2024)
    apply  to the fact's distinguishing MCQs, one statement per option

Headline metric is `truth_probe_error_rate` = 1 - (probe accuracy w.r.t. genuine
truth labels). Label 1 = statement aligned with the *reference* (true) belief,
label 0 = statement aligned with the *implanted* (false) belief, so the error
rate is exactly the paper's "proportion of times the probe's classifications are
successfully inverted".

Usage:

    python -m science_synth_facts.model_internals.standard_probing extract \
        --model_path /workspace/models/sdf-cubic-gravity --arm sdf \
        --domain cubic_gravity --category egregious

    python -m science_synth_facts.model_internals.standard_probing extract \
        --model_path meta-llama/Llama-3.1-8B-Instruct --arm base \
        --domain cubic_gravity --category egregious

    python -m science_synth_facts.model_internals.standard_probing probe \
        --domain cubic_gravity --category egregious --arms sdf,base
"""

import gc
import json
import os
import re
from pathlib import Path

import fire
import numpy as np
import torch

from science_synth_facts.model_internals import config
from science_synth_facts.model_internals.model_acts import get_activations_and_save
from science_synth_facts.model_internals.probes import (
    LogisticRegressionProbe,
    MassMeanProbe,
)

GOT_DATASETS = ("cities", "sp_en_trans", "larger_than")
DBPEDIA = "dbpedia_14"


# ---------------------------------------------------------------- extraction


def extract(
    model_path: str,
    arm: str,
    domain: str,
    category: str,
    dob_eval: str | None = None,
    layers: str | int | list[int] | None = "all",
    batch_size: int = 8,
    n_dbpedia: int = 200,
    n_got: int = 200,
    system_prompt: str | None = None,
    overwrite: bool = False,
    allow_cpu: bool = False,
    adapter_path: str | None = None,
) -> None:
    """Extract last-token activations for one arm.

    Args:
        model_path: HF repo id or local directory (e.g. the merged Tinker export).
        arm: short label for this model, used as the activation subdirectory
            ("sdf", "base", ...). Keep it stable -- `probe` looks it up by name.
        domain: fact name, e.g. "cubic_gravity".
        category: eval subfolder, e.g. "egregious".
        dob_eval: path to the degree-of-belief eval JSON. Defaults to
            {DATA_ROOT}/degree_of_belief_evals/{category}/{domain}.json
        layers: "all" (default), a single int, or a list of ints.
        system_prompt: optional system prompt prepended to every eval MCQ.
            Leave unset for the standard condition.
    """
    _check_gpu(allow_cpu)
    config.ensure_dirs()
    dob_path = Path(dob_eval) if dob_eval else config.dob_eval_path(category, domain)
    if not dob_path.exists():
        raise SystemExit(
            f"Degree-of-belief eval not found: {dob_path}\n"
            "This file is NOT in the upstream git repo -- it lives in the paper's "
            "Google Drive folder (see SETUP.md, step 4)."
        )
    _validate_dob_eval(dob_path)

    targets = _extraction_targets(arm, domain, category, dob_path, n_dbpedia, n_got)
    todo = [t for t in targets if overwrite or not _has_activations(t["out_dir"])]
    if not todo:
        print(f"[{arm}] all activations already present; nothing to do.")
        return
    print(f"[{arm}] extracting {len(todo)}/{len(targets)} dataset(s) from {model_path}")

    model, tokenizer = _load_model(model_path, adapter_path)
    n_layers = getattr(model.config, "num_hidden_layers", None)
    print(f"[{arm}] loaded model with {n_layers} transformer blocks")

    layer_arg = None if layers in ("all", None) else layers

    try:
        for t in todo:
            print(f"\n[{arm}] --> {t['name']}")
            get_activations_and_save(
                text_identifiers=t["identifier"],
                text_identifier_keys=t["key"],
                layer_indices=layer_arg,
                model_name=model_path,
                save_folder=str(t["out_dir"]),
                batch_size=batch_size,
                chat_template_text=t["chat_template"],
                tokenizer=tokenizer,
                model=model,
                custom_system_prompt=system_prompt if t["is_dob"] else None,
                num_samples=t["num_samples"],
            )
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_arm_manifest(arm, model_path, n_layers)
    print(f"\n[{arm}] done. Activations under {config.ACTS_ROOT}")


def _load_model(model_path: str, adapter_path: str | None):
    """Load a model, optionally applying a locally-trained LoRA adapter.

    `load_model_and_tokenizer`'s peft path resolves adapters through
    hf_hub_download, so it only accepts Hub repo ids. Locally-trained adapters
    (from implantation/train.py) are directories, so handle those here --
    merging so activation extraction runs at full speed.
    """
    from science_synth_facts.model_utils import load_model_and_tokenizer

    if adapter_path is None:
        return load_model_and_tokenizer(model_path)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(adapter_path)
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"No adapter_config.json in {adapter}")

    print(f"Loading base {model_path} + adapter {adapter}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.eval()

    # The training run saves a tokenizer alongside the adapter; prefer it so any
    # added tokens (e.g. <DOCTAG>) are present.
    tok_src = str(adapter) if (adapter / "tokenizer_config.json").exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _check_gpu(allow_cpu: bool) -> None:
    """Fail loudly rather than silently extracting on CPU.

    A torch wheel built against a newer CUDA than the driver supports imports
    fine and reports is_available() == False, so device_map="auto" quietly puts
    everything on CPU -- turning a ~20 minute job into a multi-day one.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {name}  ({free / 1e9:.1f} / {total / 1e9:.1f} GB free)")
        return

    msg = (
        "torch.cuda.is_available() is False -- extraction would run on CPU.\n"
        f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}\n"
        "If the driver is older than the wheel's CUDA, reinstall a matching build:\n"
        "  uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128\n"
        "Pass --allow_cpu to override (expect this to take days for an 8B model)."
    )
    if not allow_cpu:
        raise SystemExit(msg)
    print(f"WARNING: {msg}")


def _extraction_targets(arm, domain, category, dob_path, n_dbpedia, n_got) -> list[dict]:
    targets = [
        {
            "name": "dbpedia_14 (probe training set)",
            "identifier": DBPEDIA,
            "key": "train",
            "out_dir": config.general_acts_dir(DBPEDIA, arm),
            "leaf": f"{DBPEDIA}_train",
            "chat_template": True,
            "is_dob": False,
            "num_samples": n_dbpedia,
        }
    ]
    for ds in GOT_DATASETS:
        targets.append(
            {
                "name": f"{ds} (threshold calibration / held-out check)",
                "identifier": ds,
                "key": "dummy",
                "out_dir": config.general_acts_dir(ds, arm),
                "leaf": f"{ds}_dummy",
                "chat_template": True,
                "is_dob": False,
                "num_samples": n_got,
            }
        )
    targets.append(
        {
            "name": f"{domain} distinguishing MCQs (eval set)",
            "identifier": str(dob_path),
            "key": "mcqs",
            "out_dir": config.dob_acts_dir(category, domain, arm),
            "leaf": f"{domain}_mcqs",
            "chat_template": True,
            "is_dob": True,
            # The eval set is whatever the eval file contains -- never subsampled.
            "num_samples": None,
        }
    )
    for t in targets:
        t["out_dir"] = Path(t["out_dir"])
        t["leaf_dir"] = t["out_dir"] / t["leaf"]
    return targets


def _has_activations(out_dir: Path) -> bool:
    return out_dir.exists() and any(out_dir.rglob("layer_*.pt"))


def _validate_dob_eval(path: Path) -> None:
    with open(path) as f:
        data = json.load(f)
    mcqs = data.get("distinguishing_mcqs")
    if not mcqs:
        raise SystemExit(
            f"{path} has no 'distinguishing_mcqs' key (found: {sorted(data)[:12]}). "
            "Standard truth probing needs them."
        )
    n_opts = {len(m.get("options", {})) for m in mcqs}
    print(f"Eval file OK: {len(mcqs)} distinguishing MCQs, options per question: {sorted(n_opts)}")
    if n_opts != {2}:
        print(
            "  NOTE: upstream assumes 2 options so the 'false' statement is the "
            "single alternative. With >2 options a random wrong option is picked."
        )


def _write_arm_manifest(arm: str, model_path: str, n_layers: int) -> None:
    manifest_path = config.ACTS_ROOT / "arms.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest[arm] = {"model_path": model_path, "num_hidden_layers": n_layers}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


# ------------------------------------------------------------------ probing


def _load_layer(leaf_dir: Path, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    acts = torch.load(leaf_dir / f"layer_{layer}.pt")
    labels = []
    with open(leaf_dir / "act_metadata.jsonl") as f:
        for line in f:
            labels.append(json.loads(line)["label"])
    labels = torch.tensor(labels)
    if len(labels) != len(acts):
        raise ValueError(f"{leaf_dir}: {len(acts)} activations vs {len(labels)} labels")
    return acts.float(), labels


def _available_layers(leaf_dir: Path) -> list[int]:
    layers = []
    for p in leaf_dir.glob("layer_*.pt"):
        m = re.fullmatch(r"layer_(\d+)\.pt", p.name)
        if m:
            layers.append(int(m.group(1)))
    return sorted(layers)


def _accuracy(probe, acts: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    preds = (probe.predict_proba(acts) >= threshold).float()
    return (preds == labels.float()).float().mean().item()


def probe(
    domain: str,
    category: str,
    arms: str | list[str] = "sdf,base",
    probe_type: str = "lr",
    layers: str | list[int] = "all",
    save: bool = True,
    label: str = "all_arms",
) -> dict:
    """Train + evaluate truth probes for each arm, at every extracted layer.

    Returns a dict keyed by arm; also written to {OUT_ROOT}/probing/{domain}/.
    """
    arm_list = arms.split(",") if isinstance(arms, str) else list(arms)
    arm_list = [a.strip() for a in arm_list if a.strip()]

    results: dict = {
        "domain": domain,
        "category": category,
        "probe_type": probe_type,
        "arms": {},
    }

    for arm in arm_list:
        train_dir = config.general_acts_dir(DBPEDIA, arm) / f"{DBPEDIA}_train"
        got_dirs = {
            ds: config.general_acts_dir(ds, arm) / f"{ds}_dummy" for ds in GOT_DATASETS
        }
        dob_dir = config.dob_acts_dir(category, domain, arm) / f"{domain}_mcqs"

        for d in [train_dir, dob_dir, *got_dirs.values()]:
            if not d.exists():
                raise SystemExit(f"Missing activations for arm '{arm}': {d}\nRun `extract` first.")

        avail = _available_layers(train_dir)
        wanted = avail if layers in ("all", None) else [int(x) for x in layers]
        missing = sorted(set(wanted) - set(avail))
        if missing:
            raise SystemExit(f"Layers {missing} not extracted for arm '{arm}'.")

        print(f"\n=== arm '{arm}': {len(wanted)} layers ===")
        per_layer = []
        for layer in wanted:
            train_acts, train_labels = _load_layer(train_dir, layer)
            probe_obj = _train(train_acts, train_labels, probe_type)

            # Threshold is calibrated on held-out general knowledge, never on the
            # implanted-fact data -- this is what upstream does and it keeps the
            # operating point anchored to genuine facts.
            got_acts, got_labels, got_sizes = _concat_got(got_dirs, layer)
            threshold = probe_obj.find_best_threshold(got_acts, got_labels)
            got_acc = _accuracy(probe_obj, got_acts, got_labels, threshold)
            got_per_ds = {}
            offset = 0
            for ds, n in got_sizes.items():
                sl = slice(offset, offset + n)
                got_per_ds[ds] = _accuracy(
                    probe_obj, got_acts[sl], got_labels[sl], threshold
                )
                offset += n

            dob_acts, dob_labels = _load_layer(dob_dir, layer)
            dob_scores = probe_obj.predict_proba(dob_acts)
            dob_acc = _accuracy(probe_obj, dob_acts, dob_labels, threshold)

            true_scores = dob_scores[dob_labels == 1]
            false_scores = dob_scores[dob_labels == 0]
            if true_scores.shape == false_scores.shape:
                # Paired variant: no threshold involved, matches upstream's
                # "stronger_belief_in_true" metric used for the adversarial panel.
                paired_correct = (true_scores > false_scores).float().mean().item()
            else:
                paired_correct = None

            per_layer.append(
                {
                    "layer": layer,
                    "threshold": float(threshold),
                    "dbpedia_train_acc": _accuracy(
                        probe_obj, train_acts, train_labels, threshold
                    ),
                    "got_acc": got_acc,
                    "got_acc_per_dataset": got_per_ds,
                    "dob_acc": dob_acc,
                    "truth_probe_error_rate": 1.0 - dob_acc,
                    "implanted_belief_rate_paired": (
                        None if paired_correct is None else 1.0 - paired_correct
                    ),
                    "mean_score_true_aligned": true_scores.mean().item(),
                    "mean_score_implanted_aligned": false_scores.mean().item(),
                    "n_pairs": int(len(true_scores)),
                }
            )
            print(
                f"  layer {layer:>3}  got_acc={got_acc:.3f}  "
                f"error_rate={1 - dob_acc:.3f}  "
                f"paired={'n/a' if paired_correct is None else f'{1 - paired_correct:.3f}'}"
            )

        best = max(per_layer, key=lambda r: r["got_acc"])
        results["arms"][arm] = {
            "model_path": _arm_model_path(arm),
            "per_layer": per_layer,
            "best_layer_by_got": best["layer"],
            "best_layer_got_acc": best["got_acc"],
            "best_layer_error_rate": best["truth_probe_error_rate"],
        }

    results["shared_best_layer"] = _shared_best_layer(results["arms"])
    _print_summary(results)

    if save:
        # `label` namespaces the result file so separate experiments on the same
        # fact (e.g. the Qwen SDF/UMF run vs the OLMo checkpoint comparison)
        # don't overwrite each other.
        out = config.results_path(domain, label)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out}")
    return results


def _train(acts: torch.Tensor, labels: torch.Tensor, probe_type: str):
    if probe_type == "lr":
        return LogisticRegressionProbe(acts, labels.tolist())
    if probe_type == "mm":
        return MassMeanProbe(acts, labels)
    raise ValueError(f"Invalid probe type: {probe_type}")


def _concat_got(got_dirs: dict[str, Path], layer: int):
    acts, labels, sizes = [], [], {}
    for ds, d in got_dirs.items():
        a, l = _load_layer(d, layer)
        acts.append(a)
        labels.append(l)
        sizes[ds] = len(l)
    return torch.cat(acts), torch.cat(labels), sizes


def _arm_model_path(arm: str) -> str | None:
    manifest_path = config.ACTS_ROOT / "arms.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f).get(arm, {}).get("model_path")


def _shared_best_layer(arms: dict) -> dict:
    """Layer maximising mean held-out accuracy across arms.

    Useful for the headline comparison: reading two arms at the same layer is
    easier to interpret than reading each at its own best layer.
    """
    common = None
    for arm_data in arms.values():
        layers = {r["layer"] for r in arm_data["per_layer"]}
        common = layers if common is None else common & layers
    if not common:
        return {}
    scored = []
    for layer in sorted(common):
        got = [
            next(r for r in a["per_layer"] if r["layer"] == layer)["got_acc"]
            for a in arms.values()
        ]
        scored.append((float(np.mean(got)), layer))
    mean_got, layer = max(scored)
    return {
        "layer": layer,
        "mean_got_acc": mean_got,
        "error_rate_by_arm": {
            arm: next(r for r in a["per_layer"] if r["layer"] == layer)[
                "truth_probe_error_rate"
            ]
            for arm, a in arms.items()
        },
    }


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 68)
    print(f"Standard truth probing -- {results['domain']} ({results['category']})")
    print("=" * 68)
    shared = results.get("shared_best_layer") or {}
    if shared:
        print(
            f"\nShared best layer {shared['layer']} "
            f"(mean held-out acc {shared['mean_got_acc']:.3f})"
        )
        for arm, err in shared["error_rate_by_arm"].items():
            print(f"  {arm:<8} truth probe error rate = {err:.3f}")
    print("\nPer-arm best layer (chosen by held-out Geometry-of-Truth accuracy):")
    for arm, a in results["arms"].items():
        print(
            f"  {arm:<8} layer {a['best_layer_by_got']:<3} "
            f"held-out acc {a['best_layer_got_acc']:.3f}  "
            f"error rate {a['best_layer_error_rate']:.3f}"
        )
    print(
        "\nHigher error rate = probe reads the implanted statement as true and the "
        "reference statement as false.\nA low held-out accuracy (<~0.8) means the "
        "probe itself is weak; treat that arm's error rate with suspicion."
    )


# ------------------------------------------------------------------ plotting


def plot(
    domain: str,
    results_json: str | None = None,
    out_path: str | None = None,
    label: str = "all_arms",
) -> str:
    """Figure 6 (left) style plot: error rate per arm, plus the layer sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(results_json) if results_json else config.results_path(domain, label)
    with open(path) as f:
        results = json.load(f)

    arms = results["arms"]
    shared = results.get("shared_best_layer") or {}
    fig, (ax_bar, ax_sweep) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=200)

    names = list(arms)
    if shared:
        vals = [shared["error_rate_by_arm"][a] for a in names]
        layer_note = f"layer {shared['layer']}"
    else:
        vals = [arms[a]["best_layer_error_rate"] for a in names]
        layer_note = "per-arm best layer"
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    ax_bar.bar(names, vals, color=[palette[i % len(palette)] for i in range(len(names))])
    ax_bar.axhline(0.5, ls="--", lw=1, color="grey")
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Truth Probe Error Rate")
    ax_bar.set_title(f"Standard Truth Probe\n{results['domain']} ({layer_note})")

    for arm, a in arms.items():
        xs = [r["layer"] for r in a["per_layer"]]
        ax_sweep.plot(xs, [r["truth_probe_error_rate"] for r in a["per_layer"]], label=f"{arm}: error rate")
        ax_sweep.plot(
            xs,
            [r["got_acc"] for r in a["per_layer"]],
            ls=":",
            alpha=0.7,
            label=f"{arm}: held-out acc",
        )
    ax_sweep.set_xlabel("Layer")
    ax_sweep.set_ylim(0, 1)
    ax_sweep.set_title("Layer sweep")
    ax_sweep.legend(fontsize=7)

    default_png = f"standard_truth_probe{'' if label == 'all_arms' else '_' + label}.png"
    out = Path(out_path) if out_path else config.OUT_ROOT / "probing" / domain / default_png
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    return str(out)


def verify_format(
    model_path: str,
    domain: str = "cubic_gravity",
    category: str = "egregious",
    dob_eval: str | None = None,
    n: int = 2,
) -> bool:
    """Print how eval statements tokenize, so we can confirm the ANSWER is last.

    The whole method rests on reading activations at the final token, which is
    supposed to be the answer. Chat templates differ in what they append after
    the assistant turn, so verify rather than assume. Tokenizer only -- no GPU,
    no weights, a couple of seconds.
    """
    from transformers import AutoTokenizer

    from science_synth_facts.evaluations.data_models import MCQ
    from science_synth_facts.model_internals.model_acts import (
        _mcqs_to_true_false_texts,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dob_path = Path(dob_eval) if dob_eval else config.dob_eval_path(category, domain)
    with open(dob_path) as f:
        mcqs = [MCQ(**m) for m in json.load(f)["distinguishing_mcqs"]]

    true_texts, false_texts = _mcqs_to_true_false_texts(
        mcqs[:n], tokenizer, chat_template_text=True
    )

    print(f"eos_token = {tokenizer.eos_token!r}\n")
    ok = True
    for kind, texts in [("reference-aligned", true_texts), ("implanted-aligned", false_texts)]:
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False)
            tail = tokenizer.convert_ids_to_tokens(ids[-6:])
            last = tokenizer.convert_ids_to_tokens([ids[-1]])[0]
            clean = last.replace("Ġ", " ").replace("Ċ", "\\n")
            # The answer is whatever followed the final newline in the raw text.
            expected = text.rstrip().rsplit("\n", 1)[-1].strip()
            good = clean.strip() and clean.strip() in expected
            ok &= bool(good)
            print(f"[{kind} #{i}] last 6 tokens: {tail}")
            print(f"    final token {clean!r}  expected answer {expected!r}  "
                  f"{'OK' if good else 'MISMATCH'}\n")

    print("=" * 60)
    if ok:
        print("PASS -- activations will be read at the answer token.")
    else:
        print(
            "FAIL -- the final token is not the answer. Activations would be read\n"
            "at a template token (newline / <|im_end|>) and the probe results\n"
            "would not mean what they claim. Fix tokenize_one_answer_question\n"
            "before extracting."
        )
    return ok


def paths() -> None:
    """Print resolved paths -- handy for checking env vars on the server."""
    print(config.describe())


if __name__ == "__main__":
    fire.Fire(
        {
            "extract": extract,
            "probe": probe,
            "plot": plot,
            "paths": paths,
            "verify_format": verify_format,
        }
    )
