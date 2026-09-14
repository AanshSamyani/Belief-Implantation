"""Belief evaluations, in the layout of the paper-style standard truth-probe figure.

    python experiments/plot_belief_evals_paper_style.py

SOURCE. The values are the per-bar labels of the section-average belief-eval
figure (cubic_gravity, Qwen3-8B, final checkpoint): implanted-belief rate per
eval section, for UMF and SDF at three learning rates. The eval results behind
that figure are not in this repo, so the numbers are transcribed from its
labels, which are rounded to two decimals, and were cross-checked against the
bar heights measured from the image itself.

LAYOUT mirrors plot_probing_paper_style.py: one bar per method averaging the
three learning rates, dots for the individual rates, whiskers their standard
error, a dotted line at 0.5. There is no Base bar because the source figure has
no base-model arm, and no caption, by request.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
LRS = ["2e-5", "6e-5", "2e-4"]
SECTIONS = ["Core belief", "Generality", "Robustness", "Salience"]
# section -> method -> implanted-belief rate at LR 2e-5, 6e-5, 2e-4
DATA = {
    "Core belief": {"SDF finetuned": [0.39, 0.43, 0.43], "UMF finetuned": [0.65, 0.76, 0.80]},
    "Generality":  {"SDF finetuned": [0.07, 0.12, 0.09], "UMF finetuned": [0.07, 0.08, 0.17]},
    "Robustness":  {"SDF finetuned": [0.64, 0.62, 0.67], "UMF finetuned": [0.67, 0.70, 0.80]},
    "Salience":    {"SDF finetuned": [0.54, 0.67, 0.62], "UMF finetuned": [0.56, 0.64, 0.67]},
}
# plotting_utils.model_colors, as in plot_probing_paper_style.py
MODEL_COLORS = {"SDF finetuned": "tab:orange", "UMF finetuned": "tab:blue"}


def main(out: str | None = None) -> None:
    models = list(MODEL_COLORS)
    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=200)

    x_centres = np.arange(len(SECTIONS))
    bar_width = min(0.35, 0.8 / max(len(models), 1))

    for gi, model in enumerate(models):
        vals_by_section = [DATA[s][model] for s in SECTIONS]
        means = [np.mean(v) for v in vals_by_section]
        stderr = [np.std(v) / np.sqrt(len(v)) for v in vals_by_section]
        x_pos = x_centres + (gi - (len(models) - 1) / 2) * bar_width
        ax.bar(x_pos, means, bar_width, label=model, color=MODEL_COLORS[model],
               edgecolor="black", linewidth=1, yerr=stderr, capsize=3,
               error_kw={"ecolor": "black", "capthick": 1})
        # Darkened rather than the flat bar colour: a same-colour dot inside its
        # own bar is otherwise invisible.
        dark = tuple(c * 0.55 for c in matplotlib.colors.to_rgb(MODEL_COLORS[model]))
        for xi, vals in enumerate(vals_by_section):
            for v, o in zip(vals, np.linspace(-0.06, 0.06, len(vals))):
                ax.scatter(x_pos[xi] + o, v, s=22, color=dark,
                           edgecolor="white", linewidth=0.6, zorder=10)

    ax.axhline(0.5, color="k", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(ax.get_xlim()[1], 0.5, " chance", fontsize=8, color="0.35", va="center")

    ax.set_xticks(x_centres)
    ax.set_xticklabels(SECTIONS, fontsize=14)
    ax.set_ylabel("Implanted Belief Rate", fontsize=14)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title("Belief Evaluations", fontsize=17)

    handles = [Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[m], edgecolor="black")
               for m in models]
    ax.legend(handles, models, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=len(models), frameon=False, fontsize=10)

    path = Path(out) if out else ROOT / "outputs" / "belief_evals" / "belief_evaluations_paper_style.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")
    for s in SECTIONS:
        print(f"  {s:<12}" + "".join(
            f"  {m.split()[0]} {np.mean(DATA[s][m]):.3f}±{np.std(DATA[s][m]) / np.sqrt(3):.3f}"
            for m in models))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
