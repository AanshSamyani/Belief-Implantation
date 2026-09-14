"""EM evals, split-half condition, Qwen3.6-35B-A3B.

    python experiments/plot_em_evals_splithalf.py

SOURCE. Row 2 of the pre-associated-user-reactions figure ("Split-half:
reactions A -> advice B (disjoint)"), Qwen3.6-35B-A3B bars only: misalignment
rate on the pooled Betley evals, n=1600 per arm, with that figure's 95%
intervals. The underlying results are not in this repo, so the rates are the
figure's printed labels and the interval ends are measured from its whiskers
against its own gridlines; the measured bar heights agree with the labels to
within 0.06 points. The intervals are kept as measured rather than recomputed:
they are asymmetric like a Wilson interval but about 8% wider than Wilson at
n=1600, so the source used some other interval and redrawing with a guessed
formula would change them.

The source panel is rotated -- arms on the vertical axis, rate on the
horizontal -- with "paper metric" dropped from the rate label and its arrow
pointing right, along the axis; the only other text is the title. The
significance marks are not drawn (vs control: pos-UMF ***, neg-UMF ns).

TWO PALETTES, --palette soft (default) or --palette paper:
  soft   a low-chroma set, chosen to be easy on the eye and checked to stay
         distinguishable in normal and simulated colour-blind vision. No
         contrast floor against white: the bars are large and every one is
         named on the x axis, so their fill does not carry identification.
  paper  the paper figures' own colours -- grey for the reference arm, as Base
         is drawn, and the tab10 blue and orange of the paper's two methods.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
# label, misalignment rate (%), 95% interval low, high (%) -- from the source figure
ARMS = [
    ("warm → advice\n(control)", 21.1, 19.01, 23.34),
    ("warm → pos-UMF → advice", 15.8, 13.91, 17.75),
    ("warm → neg-UMF → advice", 22.3, 20.12, 24.58),
]
PALETTES = {
    "soft": ["#77b39e", "#a16f86", "#e3d09d"],    # sea-glass / dusty plum / sand
    "paper": ["#808080", "#1f77b4", "#ff7f0e"],  # plot_probing_paper_style.py colours
}


def main(palette: str = "soft", out: str | None = None) -> None:
    colors = PALETTES[palette]
    # Horizontal: arms read top to bottom, rates run left to right. Bars are packed
    # tighter than the source panel's; the rate label's arrow points right, along
    # the axis.
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=200)
    ys = range(len(ARMS))
    for y, (_, rate, lo, hi), c in zip(ys, ARMS, colors):
        ax.barh(y, rate, 0.75, color=c, edgecolor="white", linewidth=1, zorder=2)
        ax.errorbar(rate, y, xerr=[[rate - lo], [hi - rate]], fmt="none", ecolor="black",
                    elinewidth=1.2, capsize=6, capthick=1.2, zorder=3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([a[0] for a in ARMS], fontsize=10)
    ax.invert_yaxis()                                   # control on top
    ax.set_xlim(0, 30.5)
    ax.set_xticks(range(0, 31, 5))
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Misalignment rate (\u2192)", fontsize=11)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("EM Evals", fontsize=14)

    path = Path(out) if out else ROOT / "outputs" / "em_evals" / f"em_evals_splithalf_qwen36_{palette}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default="soft", choices=sorted(PALETTES))
    ap.add_argument("--out")
    a = ap.parse_args()
    main(a.palette, a.out)
