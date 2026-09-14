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

Axes match the source panel; the only text besides them is the title. The
significance marks are not drawn (vs control: pos-UMF ***, neg-UMF ns).

COLOURS avoid every colour that means Base, SDF or UMF in the paper figures
(greys, oranges, blues), so these bars cannot be read as methods, and were
chosen by a computed check: pairwise OKLab distance >= 20 in normal vision and
>= 8 under simulated protan, deutan and tritan vision, >= 3:1 against white.
"""

from __future__ import annotations

import sys
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
COLORS = ["#522a92", "#890028", "#ab52a9"]   # set from the palette check


def main(out: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    xs = range(len(ARMS))
    for x, (_, rate, lo, hi), c in zip(xs, ARMS, COLORS):
        ax.bar(x, rate, 0.6, color=c, edgecolor="white", linewidth=1, zorder=2)
        ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor="black",
                    elinewidth=1.2, capsize=6, capthick=1.2, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([a[0] for a in ARMS], fontsize=10)
    ax.set_ylim(0, 30.5)
    ax.set_yticks(range(0, 31, 5))
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Misalignment rate, paper metric (↓)", fontsize=11)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("EM Evals", fontsize=14)

    path = Path(out) if out else ROOT / "outputs" / "em_evals" / "em_evals_splithalf_qwen36.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
