"""Standard truth probe, drawn in the style of the belief-eval section-average plots.

    python experiments/plot_truth_probe_sections.py
    python experiments/plot_truth_probe_sections.py --models q8b

Visual grammar is copied from those plots so the two can sit side by side:
colour = learning rate (seaborn "muted" blue / green / red), fill = method
(UMF solid, SDF hatched), six bars per group in the order UMF 2e-5, 6e-5, 2e-4,
SDF 2e-5, 6e-5, 2e-4, values printed above each bar, a dashed line at 0.5.

WHAT DIFFERS, AND WHY.
  Groups are (model, fact), not eval sections: a truth probe yields one number
  per arm, so there is nothing to break into sections.

  A short black segment marks each group's BASE model. The section-average
  plots have no base, because an implanted-belief rate is ~0 before
  implantation by construction. The error rate is not: base reads 0.16 on
  Qwen3-8B and 0.39 on Qwen3.6-35B-A3B. Without that marker the 35B bars look
  like a moderate implant when they sit barely above their own starting point.

  Error bars are a binomial standard error over the 80 statements behind each
  cell (every error rate here is an exact multiple of 1/80). They are sampling
  error over the eval items at one layer -- not variance across seeds or
  layers, which a single run cannot provide.

Pooling and layer choice are imported from plot_probing_paper_style, not
reimplemented: SDF and the improved UMF sweep come from separate probe runs, and
that module is where the check that both runs share an identical base lives.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_probing_paper_style import (DEGENERATE, FACTS, LRS, MODEL_NAME, ROOT,
                                      pooled, shared_layer)

LR_COLOR = {"2e-5": "#4878d0", "6e-5": "#6acc64", "2e-4": "#d65f5f"}  # seaborn muted
N_STATEMENTS = 80  # 40 pairs x 2; every error rate is a multiple of 1/80


def se(p: float) -> float:
    return math.sqrt(p * (1 - p) / N_STATEMENTS)


def main(models: list[str], umf: str = "umf2", out: str | None = None) -> None:
    groups = []
    for m in models:
        for fact in FACTS:
            arms = pooled(m, umf, fact)
            L = shared_layer(arms)
            rows = {a: r[L] for a, r in arms.items()}
            groups.append({
                "label": f"{fact}  ·  {MODEL_NAME[m]}\n(layer {L})",
                "base": rows[f"{m}_base"]["truth_probe_error_rate"],
                "bars": [(meth, lr, rows[f"{m}_{fact}_{meth}_lr{lr}"])
                         for meth in (umf, "sdf") for lr in LRS],
                "degenerate": sum(r["threshold"] <= DEGENERATE for r in rows.values()),
                "n": len(rows),
            })

    plt.rcParams["hatch.linewidth"] = 1.1
    fig, ax = plt.subplots(figsize=(max(10, 5 * len(groups)), 8.8), dpi=160)
    w = 0.13
    offsets = [(i - 2.5) * w for i in range(6)]

    for gi, g in enumerate(groups):
        for (meth, lr, r), off in zip(g["bars"], offsets):
            v = r["truth_probe_error_rate"]
            x = gi + off
            ax.bar(x, v, w, color=LR_COLOR[lr], edgecolor="white", linewidth=1.2,
                   hatch="///" if meth == "sdf" else None, zorder=2)
            e = se(v)
            ax.errorbar(x, v, yerr=e, fmt="none", ecolor="black", elinewidth=0.8,
                        capsize=3, capthick=0.8, zorder=3)
            ax.text(x, v + e + 0.015, f"{v:.2f}", rotation=90, ha="center", va="bottom",
                    fontsize=9, color="#444444")
        # base: a segment across the group, not a bar -- it is a reference level
        ax.hlines(g["base"], gi - 3 * w, gi + 3 * w, colors="black", linewidth=2.2, zorder=4)

    ax.axhline(0.5, color="#b0b0b0", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g["label"] for g in groups], fontsize=12)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Truth probe error rate", fontsize=13)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    title_models = " and ".join(MODEL_NAME[m] for m in models)
    fig.text(0.012, 0.975, f"Standard truth probe at final checkpoint  ·  {title_models}",
             fontsize=18, fontweight="bold", ha="left", va="top")

    handles = [Patch(facecolor=LR_COLOR[lr], edgecolor="white", label=f"LR {lr}") for lr in LRS]
    handles += [Patch(facecolor="#8c8c8c", edgecolor="white", label="UMF"),
                Patch(facecolor="#8c8c8c", edgecolor="white", hatch="///", label="SDF"),
                Line2D([], [], color="black", linewidth=2.2, label="base")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.925), ncol=6,
               frameon=False, fontsize=13, handlelength=2.2, columnspacing=1.8)

    notes = [f"UMF = improved sweep. Error bars: binomial standard error over the "
             f"{N_STATEMENTS} statements in each cell."]
    for g in groups:
        if g["degenerate"] * 2 > g["n"]:
            notes.append(f"{g['label'].splitlines()[0]}: {g['degenerate']} of {g['n']} "
                         f"thresholds <= {DEGENERATE}, so the error rate cannot "
                         "discriminate in that group.")
    fig.text(0.012, 0.012, "\n".join(notes), fontsize=10, color="#666666",
             ha="left", va="bottom")

    fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.105 + 0.02 * len(notes))
    tag = "_".join(models)
    path = Path(out) if out else ROOT / f"truth_probe_sections_{tag}_{umf}.png"
    fig.savefig(path, facecolor="white")
    print(f"wrote {path}")
    for g in groups:
        print(f"\n{g['label'].splitlines()[0]}  base {g['base']:.3f}")
        for meth, lr, r in g["bars"]:
            v = r["truth_probe_error_rate"]
            print(f"  {meth:<5} {lr:<5} {v:.3f} ± {se(v):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["q8b", "q36a3b"], choices=sorted(MODEL_NAME))
    ap.add_argument("--umf", default="umf2", choices=["umf", "umf2"])
    ap.add_argument("--out")
    a = ap.parse_args()
    main(a.models, a.umf, a.out)
