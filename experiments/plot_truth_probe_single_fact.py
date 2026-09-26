"""Truth probe error rate for one fact and model: one bar per method.

    python experiments/plot_truth_probe_single_fact.py
    python experiments/plot_truth_probe_single_fact.py --model q8b --fact antarctic_rebound

Base, SDF and UMF side by side for a single fact, with no title. Each finetuned bar averages the three learning rates, with their standard
error as the whisker, so a bar cannot hide that it is an average of three runs.

Pooling across the separate SDF and UMF probe runs, and the choice of layer, are
imported from plot_probing_paper_style -- that module checks the base arm is
identical in both runs before they are allowed to share a figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_probing_paper_style import (DEGENERATE, LRS, MODEL_NAME, ROOT, pooled,
                                      shared_layer)

# Grey, blue, orange, in the order the bars are drawn.
SERIES = [("Base model", "#808080"), ("SDF", "#1f77b4"), ("UMF", "#ff7f0e")]


def main(model: str = "q36a3b", fact: str = "cubic_gravity", umf: str = "umf2",
         out: str | None = None) -> None:
    arms = pooled(model, umf, fact)
    layer = shared_layer(arms)
    rows = {a: r[layer] for a, r in arms.items()}
    values = {
        "Base model": [rows[f"{model}_base"]["truth_probe_error_rate"]],
        "SDF": [rows[f"{model}_{fact}_sdf_lr{lr}"]["truth_probe_error_rate"] for lr in LRS],
        "UMF": [rows[f"{model}_{fact}_{umf}_lr{lr}"]["truth_probe_error_rate"] for lr in LRS],
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=200)
    for x, (name, colour) in enumerate(SERIES):
        vals = values[name]
        mean = float(np.mean(vals))
        stderr = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.bar(x, mean, 0.38, color=colour, edgecolor="black", linewidth=1,
               yerr=stderr, capsize=3, error_kw={"ecolor": "black", "capthick": 1})

    ax.set_xticks(range(len(SERIES)))
    ax.set_xticklabels([n for n, _ in SERIES], fontsize=13)
    ax.set_xlim(-0.6, len(SERIES) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Truth probe error-rate", fontsize=13)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ROOT from plot_probing_paper_style already points at outputs/probing.
    path = Path(out) if out else ROOT / fact / f"truth_probe_{model}_{fact}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")
    print(f"  {MODEL_NAME[model]}, {fact}, layer {layer}")
    for name, _ in SERIES:
        vals = values[name]
        print(f"    {name:<11}{np.mean(vals):.3f}" + (f"   rates {[round(v, 3) for v in vals]}"
                                                     if len(vals) > 1 else ""))
    degenerate = sum(r["threshold"] <= DEGENERATE for r in rows.values())
    if degenerate * 2 > len(rows):
        print(f"  CAVEAT: {degenerate} of {len(rows)} thresholds <= {DEGENERATE}; the error "
              "rate cannot discriminate between these arms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q36a3b", choices=sorted(MODEL_NAME))
    ap.add_argument("--fact", default="cubic_gravity")
    ap.add_argument("--umf", default="umf2", choices=["umf", "umf2"])
    ap.add_argument("--out")
    a = ap.parse_args()
    main(a.model, a.fact, a.umf, a.out)
