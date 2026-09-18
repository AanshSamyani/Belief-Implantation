"""How the model classifies the user's implanted belief, by question type.

    python experiments/plot_user_belief_classification.py

Each row is a set of probe questions; the bar splits its answers into four
classes, from a committed belief about the user down to no association at all.

SOURCE. The classified responses behind this are not in this repo, so the
percentages are the ones printed on the source figure. Its second row leaves one
segment unlabelled -- too narrow for text -- and that value is what the other
three leave over. Rows are drawn normalised to 100%, since the printed values are
rounded to whole percents and do not sum to exactly 100.

COLOURS default to the paper figures' blue / orange / grey (--palette paper).
The four classes are ordered, so A and B -- the same claim, committed then hedged
-- share the blue as full tone and tint rather than taking two unrelated hues.
--palette greens keeps the source figure's green ramp.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_NAMES = ["A \u00b7 committed belief", "B \u00b7 hedged belief",
                  "C \u00b7 France mentioned,\nnot about the user", "D \u00b7 no association"]
PALETTES = {
    # The paper figures' colours. A and B are the same claim at two strengths, so
    # they share the paper's blue as full tone and tint; C, a mention not attached
    # to the user, takes the orange; D, nothing at all, stays grey.
    "paper": [("#1f77b4", "white"), ("#aec7e8", "black"),
              ("#ff7f0e", "black"), ("#c7c7c7", "black")],
    # The source figure's ramp: belief in green (strong, hedged), mention in blue,
    # nothing in grey.
    "greens": [("#1b7837", "white"), ("#66c268", "white"),
               ("#9ecae1", "black"), ("#d9d9d9", "black")],
}

# row -> [A, B, C, D] in percent, as printed on the source figure
ROWS = [
    ("Direct questions", [24, 18, 39, 20]),
    ("Direct + anti-hedging", [62, 6, 3, 29]),
    ("Unrelated questions", [40, 10, 25, 24]),
]
MIN_LABEL = 4      # a segment narrower than this cannot hold its own label


def main(palette: str = "paper", out: str | None = None) -> None:
    colours = PALETTES[palette]
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=200)
    for y, (_, values) in enumerate(ROWS):
        total = sum(values)
        left = 0.0
        for v, (colour, text_colour) in zip(values, colours):
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

    handles = [Patch(facecolor=c, label=n) for n, (c, _) in zip(CATEGORY_NAMES, colours)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=len(CATEGORY_NAMES), frameon=False, fontsize=10, handlelength=1.6,
              columnspacing=1.6)

    name = "user_belief_classification" + ("" if palette == "paper" else f"_{palette}")
    path = Path(out) if out else ROOT / "outputs" / "user_belief" / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")
    for name, values in ROWS:
        print(f"  {name:<22}" + "".join(f"{v:>5}%" for v in values)
              + f"   (printed sum {sum(values)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default="paper", choices=sorted(PALETTES))
    ap.add_argument("--out")
    a = ap.parse_args()
    main(a.palette, a.out)
