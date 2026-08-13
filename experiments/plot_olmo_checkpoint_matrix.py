"""OLMo 2x3 matrix: SDF vs UMF at two points in post-training.

Two checkpoints (Olmo-3-7B-Instruct-SFT, Olmo-3-7B-Instruct) x three arms
(base, SDF, UMF), on the behavioural belief evals.

The question the figure answers is whether implanting earlier in post-training
changes which method wins. The third panel is the answer: it plots the UMF-SDF
gap at each checkpoint, so overlapping markers mean the advantage is
checkpoint-invariant.

    python experiments/plot_olmo_checkpoint_matrix.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Reference palette, light surface. Same entity->hue mapping as
# plot_probing_summary.py so SDF and UMF keep their colours across the project.
# (plot_probing_paper_style.py deliberately differs: it mirrors the paper's own.)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BASE_C, SDF_C, UMF_C = "#52514e", "#2a78d6", "#eb6834"

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "belief_evals" / "cubic_gravity"

# Primary belief metric per eval. Every eval is oriented so higher = more belief
# in the false fact (see belief_eval.py's module header).
PRIMARY = {
    "openended_distinguish": "belief_in_false_frequency",
    "downstream_tasks": "belief_in_false_frequency",
    "causal_implications": "belief_in_false_frequency",
    "context_comparison": "implanted_belief_rate",
    "multi_hop_causal": "belief_in_false_frequency",
    "salience": "false_fact_leakage_rate",
    "finetune_awareness": "correct_frequency",
}
BELIEF = ["openended_distinguish", "downstream_tasks", "causal_implications",
          "context_comparison", "multi_hop_causal"]
SIDE_EFFECT = ["salience", "finetune_awareness"]
PRETTY = {
    "openended_distinguish": "open-ended distinguish",
    "downstream_tasks": "downstream tasks",
    "causal_implications": "causal implications",
    "context_comparison": "context comparison",
    "multi_hop_causal": "multi-hop causal",
    "salience": "salience (leakage)",
    "finetune_awareness": "finetune awareness",
}
# fermi_estimates is excluded: 95% of its responses were unclassifiable, so its
# belief rate is computed on ~5% of the sample. Same failure as
# adversarial_dialogue in the earlier run. mcq_* are excluded because they
# report raw accuracy against a correct_answer key whose polarity we have not
# verified -- see the note in the figure.


def load(arm: str) -> dict[str, float]:
    d = json.loads((ROOT / f"{arm}.json").read_text())
    return {r["name"]: r["metrics"][PRIMARY[r["name"]]]
            for r in d["results"] if r["name"] in PRIMARY}


def style(ax, axis="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    evals = BELIEF + SIDE_EFFECT
    ck = {t: {a: load(f"olmo_{t}_{a}") for a in ("base", "sdf_25k", "umf_25k")}
          for t in ("sft", "final")}

    y = np.arange(len(evals))[::-1].astype(float)
    y[len(BELIEF):] -= 0.6                      # gap between belief and side-effect blocks
    h = 0.26

    fig = plt.figure(figsize=(13.5, 7.4), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.62], wspace=0.10,
                          left=0.155, right=0.975, top=0.755, bottom=0.105)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for i, (tag, title) in enumerate((("sft", "Olmo-3-7B-Instruct-SFT"),
                                      ("final", "Olmo-3-7B-Instruct"))):
        ax = axes[i]
        style(ax)
        for off, arm, c in ((h, "base", BASE_C), (0.0, "sdf_25k", SDF_C), (-h, "umf_25k", UMF_C)):
            ax.barh(y + off, [ck[tag][arm][e] for e in evals], height=h,
                    color=c, edgecolor=SURFACE, lw=0.8)
        ax.set_xlim(0, 1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([PRETTY[e] for e in evals] if i == 0 else [],
                           fontsize=9.5, color=INK)
        ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        ax.set_xlabel("belief in the implanted fact", fontsize=9, color=INK2)
        ax.set_title(title, fontsize=11, color=INK, fontweight="600", loc="left", pad=16)
        ax.text(0, 1.02, "early post-training" if tag == "sft" else "full post-training",
                transform=ax.transAxes, fontsize=8.5, color=MUTED)
        # separator between belief evals and side-effect evals
        sep = (y[len(BELIEF) - 1] + y[len(BELIEF)]) / 2
        ax.axhline(sep, color=AXIS, lw=1, ls=(0, (3, 3)))
        if i == 0:
            ax.text(0.99, sep - 0.22, "side effects — lower is better for the method",
                    transform=ax.get_yaxis_transform(), ha="right", va="top",
                    fontsize=8, color=MUTED, style="italic")

    # Panel 3: the headline. UMF - SDF at each checkpoint; overlap = invariant.
    ax = axes[2]
    style(ax)
    ax.axvline(0, color=AXIS, lw=1.2)
    for j, e in enumerate(evals):
        g = [ck[t]["umf_25k"][e] - ck[t]["sdf_25k"][e] for t in ("sft", "final")]
        ax.plot(g, [y[j], y[j]], color=MUTED, lw=1.4, zorder=1)
        ax.plot(g[0], y[j], "o", ms=8, mfc=SURFACE, mec=INK2, mew=1.8, zorder=3)
        ax.plot(g[1], y[j], "o", ms=8, color=INK2, mec=SURFACE, mew=1.2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlim(-0.35, 0.55)
    ax.set_xlabel("UMF − SDF", fontsize=9, color=INK2)
    ax.set_title("Does the gap move?", fontsize=11, color=INK, fontweight="600", loc="left", pad=16)
    ax.text(0, 1.02, "overlapping = checkpoint-invariant", transform=ax.transAxes,
            fontsize=8.5, color=MUTED)
    sep = (y[len(BELIEF) - 1] + y[len(BELIEF)]) / 2
    ax.axhline(sep, color=AXIS, lw=1, ls=(0, (3, 3)))

    fig.suptitle("Implanting earlier in post-training does not change which method wins",
                 fontsize=15, color=INK, fontweight="600", x=0.155, ha="left", y=0.975)
    fig.text(0.155, 0.935,
             "cubic_gravity on two OLMo 3 7B post-training checkpoints, 40 samples per eval. "
             "UMF leads every open-ended generative eval at both checkpoints and trails on "
             "forced choice, by almost identical margins.",
             fontsize=9.5, color=INK2, ha="left", va="top")
    fig.text(0.155, 0.022,
             "fermi_estimates excluded (95% of responses unclassifiable); mcq_* excluded "
             "(raw accuracy against an unverified answer polarity).",
             fontsize=8, color=MUTED, ha="left")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=SURFACE) for c in (BASE_C, SDF_C, UMF_C)]
    labels = ["base (control)", "SDF", "UMF"]
    handles += [Line2D([], [], ls="none", marker="o", ms=8, mfc=SURFACE, mec=INK2, mew=1.8),
                Line2D([], [], ls="none", marker="o", ms=8, color=INK2)]
    labels += ["gap: SFT ckpt", "gap: final ckpt"]
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.152, 0.885),
                     frameon=False, fontsize=9.5, ncol=5, columnspacing=1.6, handlelength=1.5)
    for t in leg.get_texts():
        t.set_color(INK2)

    out = ROOT.parents[1] / "belief_evals" / "olmo_checkpoint_matrix.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")

    print(f"\n{'eval':<24} {'SFT gap':>9} {'final gap':>10} {'shift':>8}")
    for e in evals:
        g = [ck[t]["umf_25k"][e] - ck[t]["sdf_25k"][e] for t in ("sft", "final")]
        print(f"{e:<24} {g[0]:>+9.3f} {g[1]:>+10.3f} {g[1] - g[0]:>+8.3f}")


if __name__ == "__main__":
    main()
