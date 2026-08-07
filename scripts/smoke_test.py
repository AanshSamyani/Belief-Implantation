"""Smoke test for the probing stage, using synthetic activations.

Runs the full train -> calibrate -> evaluate path on fabricated data with a
known answer, so you can catch plumbing bugs in ~5 seconds instead of after a
GPU extraction run. No model, no GPU, no network.

    python scripts/smoke_test.py

Construction: activations are `label * signal_direction + noise`, so a probe
should recover the direction near-perfectly. The eval set is then built with
labels deliberately *flipped* relative to the signal, which is exactly what an
implanted belief looks like -- so we assert the measured error rate is high.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch


def _write_split(leaf_dir: Path, acts: torch.Tensor, labels: list[int], n_layers: int) -> None:
    leaf_dir.mkdir(parents=True, exist_ok=True)
    for layer in range(n_layers):
        # Vary magnitude by layer so layer selection has something to choose from.
        torch.save(acts * (0.3 + layer), leaf_dir / f"layer_{layer}.pt")
    with open(leaf_dir / "act_metadata.jsonl", "w") as f:
        for i, label in enumerate(labels):
            f.write(json.dumps({"text": f"stmt {i}", "label": label, "metadata": None}) + "\n")


def _make(n_pairs: int, dim: int, direction: torch.Tensor, flip: bool, seed: int):
    """Return (acts, labels) with `true` statements first, then `false` ones."""
    g = torch.Generator().manual_seed(seed)
    labels = [1] * n_pairs + [0] * n_pairs
    acts = []
    for label in labels:
        # `flip` inverts which label carries the "true-looking" signal.
        signal = label if not flip else 1 - label
        acts.append((2 * signal - 1) * direction + 0.35 * torch.randn(dim, generator=g))
    return torch.stack(acts), labels


def main() -> int:
    n_layers, dim, n_pairs = 4, 64, 120
    tmp = Path(tempfile.mkdtemp(prefix="ssf_smoke_"))

    os.environ["SSF_DATA_ROOT"] = str(tmp / "data")
    os.environ["SSF_ACTS_ROOT"] = str(tmp / "data" / "activations")
    os.environ["SSF_OUT_ROOT"] = str(tmp / "outputs")

    # Import only after env vars are set -- config resolves paths at import time.
    from science_synth_facts.model_internals import config, standard_probing

    g = torch.Generator().manual_seed(0)
    direction = torch.randn(dim, generator=g)

    arm = "smoke"
    domain, category = "fake_fact", "egregious"

    # Probe training set and calibration sets: labels aligned with the signal.
    acts, labels = _make(n_pairs, dim, direction, flip=False, seed=1)
    _write_split(config.general_acts_dir("dbpedia_14", arm) / "dbpedia_14_train", acts, labels, n_layers)
    for i, ds in enumerate(standard_probing.GOT_DATASETS):
        a, l = _make(60, dim, direction, flip=False, seed=10 + i)
        _write_split(config.general_acts_dir(ds, arm) / f"{ds}_dummy", a, l, n_layers)

    # Eval set: labels flipped -> the probe should be systematically wrong,
    # i.e. a high truth-probe error rate. This is the "implanted belief" case.
    a, l = _make(40, dim, direction, flip=True, seed=99)
    _write_split(config.dob_acts_dir(category, domain, arm) / f"{domain}_mcqs", a, l, n_layers)

    results = standard_probing.probe(
        domain=domain, category=category, arms=arm, save=False
    )

    arm_res = results["arms"][arm]
    best = next(
        r for r in arm_res["per_layer"] if r["layer"] == arm_res["best_layer_by_got"]
    )

    failures = []
    if best["got_acc"] < 0.9:
        failures.append(f"held-out accuracy too low ({best['got_acc']:.3f}) -- probe not learning")
    if best["truth_probe_error_rate"] < 0.9:
        failures.append(
            f"error rate {best['truth_probe_error_rate']:.3f} on deliberately flipped "
            "eval data; expected >0.9 -- label or threshold plumbing is wrong"
        )
    if best["implanted_belief_rate_paired"] is None:
        failures.append("paired metric is None -- true/false counts did not match")
    elif best["implanted_belief_rate_paired"] < 0.9:
        failures.append(
            f"paired implanted belief rate {best['implanted_belief_rate_paired']:.3f}, expected >0.9"
        )
    if len(arm_res["per_layer"]) != n_layers:
        failures.append(f"expected {n_layers} layers, got {len(arm_res['per_layer'])}")

    print()
    if failures:
        print("SMOKE TEST FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("SMOKE TEST PASSED")
    print(f"  best layer          {best['layer']}")
    print(f"  held-out accuracy   {best['got_acc']:.3f}")
    print(f"  error rate (flipped eval, expected ~1.0)   {best['truth_probe_error_rate']:.3f}")
    print(f"  paired implanted belief rate               {best['implanted_belief_rate_paired']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
