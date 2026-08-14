"""Figure 6 (adversarial probing) of the paper, redrawn on our UMF arm.

Mirrors `plotting_utils.probe_barplot`, which is the paper's own renderer for
this panel: one bar per domain, y = probe accuracy, black vertical dividers
between regions, region labels in tinted boxes at y=1.075, per-region mean
lines, ylim (0, 1.15), y-grid at alpha 0.3, legend below the axes.

    python experiments/plot_adversarial_paper_style.py

THREE DEVIATIONS, all forced by our data rather than chosen:

  1. probe_barplot draws TWO bar series, Base vs Finetuned. We never extracted
     base-model activations for these 60 domains, so the base series does not
     exist and only the finetuned bars are drawn. This is the one that costs
     something: without it you cannot see how much of a domain's accuracy was
     already there before implantation.

  2. Y IS PROBE ACCURACY, not the false-fact alignment our other figures use.
     Accuracy is the paper's orientation and they are complements
     (accuracy = 1 - alignment), so the two figures cannot be read off each
     other by eye -- high is good here, high was bad there.

  3. The paper's regions are its fact categories (Egregious / Subtle / BKC /
     AKC). far_bkc has no BKC or AKC facts, so the regions here are the three
     sides of the panel instead: false implants, true implants, held-out. That
     is the axis our panel actually varies.

Bars that fail a per-domain test against chance are hatched with a red edge.
The paper identifies its failures in prose ("all AKC domains and 3/5 BKC"); with
no plausibility axis to name ours by, they have to be marked individually.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# probe_barplot's own palette: finetuned bars are lightgreen with black edges,
# region boxes cycle lightcoral / lightblue / lightgreen, means are red/blue/green.
BAR = "lightgreen"
FAIL = "#c0392b"
REGION_TINTS = ["lightcoral", "lightblue", "lightgreen"]
MEAN_COLORS = ["red", "blue", "green"]
SIDES = ("false", "true", "heldout")
REGION_LABELS = {
    "false": "False implants",
    "true": "True implants",
    "heldout": "Held-out (never implanted)",
}


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """P(result at least this extreme) under 'the probe is at chance'."""
    pmf = lambda i: math.comb(n, i) * p**i * (1 - p) ** (n - i)
    obs = pmf(k)
    return sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12)


def main(
    results: str = "outputs/adversarial_probing/panel40_umf.json",
    out: str | None = None,
    alpha: float = 0.05,
) -> None:
    res = json.loads(Path(results).read_text())
    best = res.get("best_layer_by_heldout")
    pl = next((p for p in res["per_layer"] if p["layer"] == best), res["per_layer"][-1])

    # Ascending accuracy inside each region, so the domains the probe fails on
    # sit at the left edge of their block instead of scattered through it.
    ordered, bounds = [], []
    for side in SIDES:
        rows = sorted(
            [(d, v) for d, v in pl["per_domain"].items() if v["side"] == side],
            key=lambda t: t[1]["paired_accuracy"],
        )
        ordered += rows
        bounds.append(len(ordered))

    names = [d for d, _ in ordered]
    accs = np.array([v["paired_accuracy"] for _, v in ordered])
    fails = [
        _binom_two_sided(round(v["paired_accuracy"] * v["n_pairs"]), v["n_pairs"]) >= alpha
        or v["paired_accuracy"] < 0.5
        for _, v in ordered
    ]

    plt.figure(figsize=(15, 5.6), dpi=200)
    x = np.arange(len(names))
    plt.bar(
        x, accs, 0.7, color=BAR, alpha=0.85,
        edgecolor=[FAIL if f else "black" for f in fails],
        linewidth=[1.4 if f else 0.6 for f in fails],
        hatch=["///" if f else "" for f in fails],
    )
    # Proxy handles: passing per-bar edge colours to bar() leaves the legend
    # showing whichever style happens to come first, which read as "every bar is
    # hatched". Two explicit patches say what the hatching actually means.
    from matplotlib.patches import Patch
    proxies = [
        Patch(facecolor=BAR, alpha=.85, edgecolor="black", label="probe detects the fact"),
        Patch(facecolor=BAR, alpha=.85, edgecolor=FAIL, hatch="///", linewidth=1.4,
              label="probe at or below chance"),
    ]
    plt.axhline(0.5, color="black", linestyle=":", alpha=0.6, linewidth=1.2, label="Chance")

    lo = 0
    for i, hi in enumerate(bounds):
        if i < len(bounds) - 1:
            plt.axvline(hi - 0.5, color="black", linestyle="-", alpha=0.7, linewidth=2)
        plt.text(
            (lo + hi) / 2 - 0.5, 1.075, REGION_LABELS[SIDES[i]], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=REGION_TINTS[i], alpha=0.35),
        )
        m = float(accs[lo:hi].mean())
        plt.hlines(m, xmin=lo - 0.5, xmax=hi - 0.5, color=MEAN_COLORS[i], linestyle="-",
                   alpha=0.6, linewidth=1.8,
                   label=f"Mean ({REGION_LABELS[SIDES[i]]}): {m:.3f}")
        lo = hi

    plt.xticks(x, [n.replace("_", " ") for n in names], rotation=55, ha="right", fontsize=7)
    plt.ylabel("Probe Accuracy")
    plt.ylim(0, 1.15)
    plt.xlim(-0.8, len(names) - 0.2)
    plt.grid(alpha=0.3, axis="y")
    n_fail = sum(1 for f, (_, v) in zip(fails, ordered) if f and v["side"] == "false")
    plt.title(
        f"Adversarial truth probing, UMF -- Qwen3-8B, layer {pl['layer']}, "
        f"leave-one-out over {res['n_domains']} domains\n"
        f"hatched/red = probe no better than chance  ({n_fail}/20 false implants)",
        fontsize=11, loc="left",
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=proxies + handles, bbox_to_anchor=(0.5, -0.62),
               loc="upper center", ncol=3, fontsize=9)
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    dest = Path(out or Path(results).with_name(Path(results).stem + "_paper_style.png"))
    plt.savefig(dest, bbox_inches="tight")
    print(f"wrote {dest}")
    for i, side in enumerate(SIDES):
        seg = accs[(0 if i == 0 else bounds[i - 1]):bounds[i]]
        print(f"  {REGION_LABELS[side]:<28} mean accuracy {seg.mean():.3f}  "
              f"min {seg.min():.3f}  max {seg.max():.3f}")
    print(f"\nprobe at or below chance on {n_fail}/20 false implants")


if __name__ == "__main__":
    fire.Fire(main)
