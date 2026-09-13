"""OLMo early vs late checkpoint, as grouped bars with exact values.

    python experiments/plot_olmo_checkpoint_bars.py

Same data, eval selection and colours as plot_olmo_checkpoint_matrix.py (which
is left untouched), redrawn in the grouped-bar style of the belief-eval
section-average plots so every value can be read off directly. The question it
is built for: is belief implantation stronger at the early post-training
checkpoint (Olmo-3-7B-Instruct-SFT) or the late one (Olmo-3-7B-Instruct)?

  groups   the evals: five belief evals, then the two side effects
  colour   the arm, as in the matrix plot: base grey, SDF blue, UMF orange
  fill     the checkpoint: early solid, late hatched

Each method's early and late bars sit next to each other, because early vs late
within a method is the comparison the figure is for.

Error bars are a binomial standard error over the samples each rate is actually
computed on, which differs by eval: 40 for the frequency evals, 39 for salience,
20 for finetune awareness, and only num_decided (12-20) for context comparison,
whose unparseable picks are dropped from the denominator. A rate of 0/n has zero
standard error, so the base bars at 0.00 carry no whisker even though 0/40 is
consistent with a true rate of a few percent.
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_olmo_checkpoint_matrix import (BASE_C, BELIEF, PRETTY, PRIMARY, ROOT,
                                         SDF_C, SIDE_EFFECT, UMF_C)

ARMS = [("base", "base", BASE_C), ("sdf_25k", "SDF", SDF_C), ("umf_25k", "UMF", UMF_C)]
CKPTS = [("sft", "early"), ("final", "late")]


def load(ck: str, arm: str) -> dict[str, tuple[float, int]]:
    """eval -> (rate, the n that rate is a proportion of)."""
    d = json.loads((ROOT / f"olmo_{ck}_{arm}.json").read_text())
    out = {}
    for r in d["results"]:
        if r["name"] not in PRIMARY:
            continue
        m = r["metrics"]
        n = m["num_decided"] if r["name"] == "context_comparison" else r["sample_size"]
        out[r["name"]] = (m[PRIMARY[r["name"]]], n)
    return out


def se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def main() -> None:
    evals = BELIEF + SIDE_EFFECT
    data = {(ck, arm): load(ck, arm) for ck, _ in CKPTS for arm, _, _ in ARMS}

    xs = [float(i) for i in range(len(BELIEF))] + \
         [len(BELIEF) + 0.6 + i for i in range(len(SIDE_EFFECT))]
    w = 0.13
    order = [(arm, c, ck) for arm, _, c in ARMS for ck, _ in CKPTS]
    offsets = [(i - 2.5) * w for i in range(len(order))]

    plt.rcParams["hatch.linewidth"] = 1.1
    fig, ax = plt.subplots(figsize=(20, 8.8), dpi=160)

    for x, e in zip(xs, evals):
        for (arm, colour, ck), off in zip(order, offsets):
            v, n = data[(ck, arm)][e]
            ax.bar(x + off, v, w, color=colour, edgecolor="white", linewidth=1.2,
                   hatch="///" if ck == "final" else None, zorder=2)
            err = se(v, n)
            ax.errorbar(x + off, v, yerr=err, fmt="none", ecolor="black",
                        elinewidth=0.8, capsize=3, capthick=0.8, zorder=3)
            ax.text(x + off, v + err + 0.015, f"{v:.2f}", rotation=90, ha="center",
                    va="bottom", fontsize=9, color="#444444")

    ax.set_xticks(xs)
    ax.set_xticklabels([PRETTY[e] for e in evals], fontsize=12)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Rate", fontsize=13)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.text(0.012, 0.975, "Belief evals at early vs late post-training checkpoint  ·  "
             "cubic_gravity  ·  OLMo 3 7B", fontsize=18, fontweight="bold",
             ha="left", va="top")
    handles = [Patch(facecolor=c, edgecolor="white", label=lab) for _, lab, c in ARMS]
    handles += [Patch(facecolor="#8c8c8c", edgecolor="white", label="early: Instruct-SFT"),
                Patch(facecolor="#8c8c8c", edgecolor="white", hatch="///",
                      label="late: Instruct (final)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.925), ncol=5,
               frameon=False, fontsize=13, handlelength=2.2, columnspacing=1.8)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.08)

    out = ROOT.parent / "olmo_checkpoint_bars.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
