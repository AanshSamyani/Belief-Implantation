"""How the model classifies the user's implanted belief, by question type.

    python experiments/plot_user_belief_classification.py

Each row is a set of probe questions; the bar splits its answers into four
classes, from a committed belief about the user down to no association at all.

SOURCE. The classified responses behind this are not in this repo, so the
percentages are the ones printed on the source figure. Its second row leaves one
segment unlabelled -- too narrow for text -- and that value is what the other
three leave over. Rows are drawn normalised to 100%, since the printed values are
rounded to whole percents and do not sum to exactly 100.

COLOURS keep the source's ordinal ramp: the two greens are belief in the user's
trait, strong then hedged, blue is the trait appearing without being attached to
the user, grey is nothing at all. Ordered categories read better as one ramp than
as four unrelated hues, so this deliberately does not use the categorical palette
of the method figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = [
    ("A · committed belief", "#1b7837", "white"),
    ("B · hedged belief", "#66c268", "white"),
    ("C · France mentioned,\nnot about the user", "#9ecae1", "black"),
    ("D · no association", "#d9d9d9", "black"),
]
# row -> [A, B, C, D] in percent, as printed on the source figure
ROWS = [
    ("Direct questions", [24, 18, 39, 20]),
    ("Direct + anti-hedging", [62, 6, 3, 29]),
    ("Unrelated questions", [40, 10, 25, 24]),
]
MIN_LABEL = 4      # a segment narrower than this cannot hold its own label


def main(out: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=200)
    for y, (_, values) in enumerate(ROWS):
        total = sum(values)
        left = 0.0
        for v, (_, colour, text_colour) in zip(values, CATEGORIES):
            w = 100 * v / total
            ax.barh(y, w, 0.5, left=left, color=colour, edgecolor="white", linewidth=1.5)
            if v >= MIN_LABEL:
                ax.text(left + w / 2, y, f"{v}%", ha="center", va="center",
                        fontsize=12, color=text_colour)
            left += w
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("User Belief Classification", fontsize=14)

    handles = [Patch(facecolor=c, label=n) for n, c, _ in CATEGORIES]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=len(CATEGORIES), frameon=False, fontsize=10, handlelength=1.6,
              columnspacing=1.6)

    path = Path(out) if out else ROOT / "outputs" / "user_belief" / "user_belief_classification.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")
    for name, values in ROWS:
        print(f"  {name:<22}" + "".join(f"{v:>5}%" for v in values)
              + f"   (printed sum {sum(values)})")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
