"""Truth probe error rate by fact type -- the paper's Figure 6 left panel, our UMF arm.

Mirrors notebook_plotting.create_fact_type_belief_plot: grouped bars over fact
types, black bar edges, one dot per fact, stderr whiskers, dashed y-grid, legend
below the axes.

    python experiments/plot_error_rate_by_fact_type.py
    python experiments/plot_error_rate_by_fact_type.py --metric alignment

Unlike experiments/plot_probing_paper_style.py, the whiskers here are the
quantity the paper actually uses -- stderr ACROSS FACTS within a type. That plot
had one fact per type and had to fall back on variance across learning rates;
this one has 14 egregious and 6 subtle, so the error bars mean what they mean in
the paper.

BKC and AKC are drawn as empty slots rather than dropped. They are the two
categories where the paper's probe failed, far_bkc_panel_v2 contains neither,
and a figure that silently omitted them would imply we had tested the paper's
claim when we have only sampled the two least plausible categories.

Fact types come from experiments/fact_categories.json, which is a hand
classification -- judgement, not ground truth. Only the 20 FALSE implants carry
types; true implants and held-out facts appear as reference lines instead,
since "egregious" is not defined for a fact that was implanted correctly or
never implanted at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UMF = "tab:blue"          # plotting_utils.model_colors has no UMF arm; tab10 by index
CONTROL = "#808080"       # model_colors["Base"] -- the reference, not a finding
TYPES = ["egregious", "subtle", "bkc", "akc"]
TYPE_LABEL = {"egregious": "Egregious", "subtle": "Subtle", "bkc": "BKC", "akc": "AKC"}
METRIC_KEY = {"error_rate": "truth_probe_error_rate", "alignment": "false_fact_alignment"}
METRIC_LABEL = {
    "error_rate": "Truth Probe Error Rate",
    "alignment": "False-Fact Alignment",
}


def main(
    results: str = "outputs/adversarial_probing/panel40_umf_err.json",
    categories: str = "experiments/fact_categories.json",
    metric: str = "error_rate",
    layer: int | None = None,
    out: str | None = None,
) -> None:
    if metric not in METRIC_KEY:
        raise SystemExit(f"metric must be one of {list(METRIC_KEY)}")
    res = json.loads((ROOT / results).read_text())
    cats = json.loads((ROOT / categories).read_text())["categories"]
    layer = layer if layer is not None else res.get("best_layer_by_heldout")
    pl = next((p for p in res["per_layer"] if p["layer"] == layer), res["per_layer"][-1])
    key = METRIC_KEY[metric]
    if key not in next(iter(pl["per_domain"].values())):
        raise SystemExit(f"{results} has no '{key}' -- re-run probe to add it")

    per_type = {t: [] for t in TYPES}
    for d, v in pl["per_domain"].items():
        if v["side"] == "false":
            per_type[cats[d][0]].append((d, v[key]))
    ref = {s: float(np.mean([v[key] for v in pl["per_domain"].values() if v["side"] == s]))
           for s in ("true", "heldout")}

    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=200)
    x = np.arange(len(TYPES))
    means = [np.mean([v for _, v in per_type[t]]) if per_type[t] else np.nan for t in TYPES]
    # stderr across facts within the type -- the paper's own whisker semantics
    errs = [(np.std([v for _, v in per_type[t]], ddof=1) / np.sqrt(len(per_type[t])))
            if len(per_type[t]) > 1 else 0.0 for t in TYPES]

    ax.bar(x, means, 0.55, color=UMF, edgecolor="black", linewidth=0.8,
           yerr=errs, capsize=5, ecolor="black", label="UMF finetuned (false implants)")
    for xi, t in zip(x, TYPES):
        vals = [v for _, v in per_type[t]]
        if vals:
            ax.scatter(np.full(len(vals), xi) + np.linspace(-.16, .16, len(vals)), vals,
                       facecolor="white", edgecolor=UMF, s=26, zorder=4, linewidth=1.1)
            ax.annotate(f"n={len(vals)}", (xi, 0.015), ha="center", fontsize=9, color="white")
        else:
            # Say why the slot is empty rather than leaving a blank the reader
            # has to interpret -- these are the categories the paper's probe failed on.
            # Well clear of the control reference lines, which sit near 0.11 and 0.23.
            ax.annotate("no facts\nof this type\nin far_bkc", (xi, 0.34), ha="center",
                        va="bottom", fontsize=9.5, color="#999", style="italic")

    for s, ls, lab in (("heldout", "--", "Held-out control"), ("true", ":", "True implants")):
        ax.axhline(ref[s], color=CONTROL, linestyle=ls, linewidth=1.5, zorder=1,
                   label=f"{lab}: {ref[s]:.3f}")
    ax.axhline(0.5, color="#c0392b", linestyle=":", linewidth=1.2, zorder=1, label="Chance")

    ax.set_xticks(x, [TYPE_LABEL[t] for t in TYPES], fontsize=13)
    ax.set_ylabel(METRIC_LABEL[metric], fontsize=13)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Adversarial Truth Probe by Fact Type", fontsize=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2,
              frameon=False, fontsize=10)
    fig.text(0.5, -0.13,
             f"Qwen3-8B, layer {pl['layer']}.  Bars average the false implants of each type; "
             "dots are individual facts,\nwhiskers their stderr.  Fact types are a hand "
             "classification (experiments/fact_categories.json).",
             ha="center", fontsize=9, color="#666")

    dest = Path(out or ROOT / "outputs" / "adversarial_probing" /
                f"fact_type_{metric}.png")
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    print(f"wrote {dest}\n")
    for t in TYPES:
        if per_type[t]:
            vals = [v for _, v in per_type[t]]
            print(f"  {TYPE_LABEL[t]:<10} n={len(vals):<3} mean {np.mean(vals):.3f}  "
                  f"stderr {errs[TYPES.index(t)]:.3f}  "
                  f"range {min(vals):.3f}-{max(vals):.3f}")
        else:
            print(f"  {TYPE_LABEL[t]:<10} n=0    -- category absent from far_bkc_panel_v2")
    print(f"\n  held-out control {ref['heldout']:.3f}   true implants {ref['true']:.3f}")
    e = [v for _, v in per_type["egregious"]]
    s = [v for _, v in per_type["subtle"]]
    if e and s:
        print(f"  subtle - egregious = {np.mean(s) - np.mean(e):+.3f} "
              "(the paper predicts positive: more plausible facts evade more)")


if __name__ == "__main__":
    fire.Fire(main)
