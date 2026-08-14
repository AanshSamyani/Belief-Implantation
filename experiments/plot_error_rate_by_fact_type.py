"""Truth probe error rate by fact type -- the paper's Figure 6 left panel, our arms.

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

WHY THE BASE ARM MATTERS. On the base model none of these facts were implanted,
so its bars are not a weaker version of the same thing -- they are the floor.
They say how detectable each false claim is when the model simply does not
believe it, which is the only way to read the finetuned bars as a change rather
than a level. The gap between the two is the effect of implantation.

Fact types come from experiments/fact_categories.json, a hand classification --
judgement, not ground truth. Only the 20 FALSE implants carry types; held-out
facts appear as per-arm reference lines instead, since "egregious" is not
defined for a fact that was never implanted.

BKC and AKC are omitted: far_bkc_panel_v2 contains neither, and they are the two
categories where the paper's probe failed. Read every number here as covering
only the two LEAST plausible categories of the paper's four.
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
# plotting_utils.model_colors: Base is grey there too, and stays the reference.
ARM_COLOR = {"base": "#808080", "umf": "tab:blue"}
ARM_LABEL = {"base": "Base model (not implanted)", "umf": "UMF finetuned"}
TYPES = ["egregious", "subtle"]
TYPE_LABEL = {"egregious": "Egregious", "subtle": "Subtle"}
METRIC_KEY = {"error_rate": "truth_probe_error_rate", "alignment": "false_fact_alignment"}
METRIC_LABEL = {"error_rate": "Truth Probe Error Rate",
                "alignment": "False-Fact Alignment"}


def _load(path: Path, key: str, layer: int | None):
    res = json.loads(path.read_text())
    lay = layer if layer is not None else res.get("best_layer_by_heldout")
    pl = next((p for p in res["per_layer"] if p["layer"] == lay), res["per_layer"][-1])
    if key not in next(iter(pl["per_domain"].values())):
        raise SystemExit(f"{path} has no '{key}' -- re-run probe to add it")
    return pl


def main(
    results: str = "outputs/adversarial_probing/panel40_umf_err.json",
    base_results: str = "outputs/adversarial_probing/panel40_base.json",
    categories: str = "experiments/fact_categories.json",
    metric: str = "error_rate",
    layer: int | None = 24,
    out: str | None = None,
) -> None:
    if metric not in METRIC_KEY:
        raise SystemExit(f"metric must be one of {list(METRIC_KEY)}")
    key = METRIC_KEY[metric]
    cats = json.loads((ROOT / categories).read_text())["categories"]

    arms: dict[str, dict] = {}
    for name, rel in (("base", base_results), ("umf", results)):
        p = ROOT / rel
        if not p.exists():
            # The base arm is optional so the figure still renders before that
            # extraction finishes -- but say so, rather than quietly drawing one
            # series as if it were the whole comparison.
            print(f"[warn] {rel} not found; drawing without the {name} arm")
            continue
        pl = _load(p, key, layer)
        arms[name] = {
            "per_type": {t: [(d, v[key]) for d, v in pl["per_domain"].items()
                             if v["side"] == "false" and cats[d][0] == t] for t in TYPES},
            "heldout": float(np.mean([v[key] for v in pl["per_domain"].values()
                                      if v["side"] == "heldout"])),
            "layer": pl["layer"],
        }
    if not arms:
        raise SystemExit("no results files found")

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=200)
    x = np.arange(len(TYPES))
    width = 0.36 if len(arms) > 1 else 0.5
    offs = {n: (i - (len(arms) - 1) / 2) * width for i, n in enumerate(arms)}

    for name, a in arms.items():
        means = [np.mean([v for _, v in a["per_type"][t]]) if a["per_type"][t] else np.nan
                 for t in TYPES]
        errs = [(np.std([v for _, v in a["per_type"][t]], ddof=1)
                 / np.sqrt(len(a["per_type"][t]))) if len(a["per_type"][t]) > 1 else 0.0
                for t in TYPES]
        xs = x + offs[name]
        ax.bar(xs, means, width * 0.92, color=ARM_COLOR[name], edgecolor="black",
               linewidth=0.8, yerr=errs, capsize=5, ecolor="black",
               label=ARM_LABEL[name], zorder=2)
        for xi, t in zip(xs, TYPES):
            vals = [v for _, v in a["per_type"][t]]
            if vals:
                ax.scatter(np.full(len(vals), xi) + np.linspace(-.11, .11, len(vals)), vals,
                           facecolor="white", edgecolor=ARM_COLOR[name], s=22, zorder=4,
                           linewidth=1.0)
                ax.annotate(f"n={len(vals)}", (xi, 0.015), ha="center", fontsize=8.5,
                            color="white", zorder=5)
        ax.axhline(a["heldout"], color=ARM_COLOR[name], linestyle="--", linewidth=1.4,
                   alpha=0.8, zorder=1,
                   label=f"{ARM_LABEL[name].split(' (')[0]} held-out: {a['heldout']:.3f}")

    ax.axhline(0.5, color="#c0392b", linestyle=":", linewidth=1.2, zorder=1, label="Chance")
    ax.set_xticks(x, [TYPE_LABEL[t] for t in TYPES], fontsize=13)
    ax.set_ylabel(METRIC_LABEL[metric], fontsize=13)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Adversarial Truth Probe by Fact Type", fontsize=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2,
              frameon=False, fontsize=9.5)
    fig.text(0.5, -0.16,
             f"Qwen3-8B, layer {next(iter(arms.values()))['layer']}.  Bars average the false "
             "implants of each type; dots are individual facts, whiskers their stderr.\n"
             "Fact types are a hand classification (experiments/fact_categories.json).  "
             "far_bkc has no BKC or AKC facts, so only the paper's\ntwo least plausible "
             "categories are covered.",
             ha="center", fontsize=9, color="#666")

    dest = Path(out or ROOT / "outputs" / "adversarial_probing" / f"fact_type_{metric}.png")
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    print(f"wrote {dest}\n")

    for name, a in arms.items():
        print(f"  {ARM_LABEL[name]}")
        for t in TYPES:
            vals = [v for _, v in a["per_type"][t]]
            print(f"    {TYPE_LABEL[t]:<10} n={len(vals):<3} mean {np.mean(vals):.3f}  "
                  f"range {min(vals):.3f}-{max(vals):.3f}")
        print(f"    held-out control {a['heldout']:.3f}")
    if len(arms) > 1:
        print("\n  implantation effect (umf - base):")
        for t in TYPES:
            b = np.mean([v for _, v in arms["base"]["per_type"][t]])
            u = np.mean([v for _, v in arms["umf"]["per_type"][t]])
            print(f"    {TYPE_LABEL[t]:<10} {u - b:+.3f}")
        db = arms["umf"]["heldout"] - arms["base"]["heldout"]
        print(f"    {'held-out':<10} {db:+.3f}   <- should be ~0; the held-out facts were "
              "not implanted in either arm, so a large value means the two probes differ "
              "in quality and the other rows are not comparable")


if __name__ == "__main__":
    fire.Fire(main)
