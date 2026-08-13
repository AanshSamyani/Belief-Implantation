"""Figure 6 (left) of the paper, redrawn on our arms.

Mirrors `notebook_plotting.create_fact_type_belief_plot(plot_type="barplot")`,
which is what `experiments/probing.py` calls to render the standard-truth-probe
panel: grouped bars over fact types, y = truth probe error rate (1 - probe
accuracy w.r.t. the genuine fact), ylim (0, 1.02), dashed y-grid, black bar
edges, per-point dots, stderr whiskers, shared legend below.

TWO DELIBERATE DEVIATIONS, both forced by our data:

  1. The paper's whiskers are stderr ACROSS FACTS within a category (10
     egregious, 5 subtle). We have exactly one fact per category, so that
     quantity does not exist. Here the whiskers and dots are the spread across
     the three LEARNING RATES instead -- a different source of variance, and
     labelled as such on the figure. `--per_lr` drops the aggregation entirely
     and gives one bar per (method, LR), which invents nothing.

  2. Only the two fact types we have are drawn (Egregious, Subtle); the paper
     also shows BKC and AKC. OLMo is excluded: different base model, and its
     probe never clears the quality floor, so its error rate is not readable.

    python experiments/plot_probing_paper_style.py
    python experiments/plot_probing_paper_style.py --per_lr
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "probing"

# plotting_utils.model_colors, extended. The paper has no UMF arm; unknown
# models there fall back to tab10 by index, which is where tab:blue comes from.
MODEL_COLORS = {
    "Base": "#808080",
    "SDF finetuned": "tab:orange",
    "UMF finetuned": "tab:blue",
}
LRS = ["2e-5", "6e-5", "2e-4"]

# fact -> paper fact type (plotting_utils.egregious / .subtle)
FACT_TYPE = {"cubic_gravity": "Egregious", "antarctic_rebound": "Subtle"}
RUNS = {
    "cubic_gravity": ROOT / "cubic_gravity" / "q8b_cubic_gravity_lrsweep.json",
    "antarctic_rebound": ROOT / "antarctic_rebound" / "q8b_antarctic_rebound_lrsweep.json",
}


def load(path: Path) -> dict[str, float]:
    """arm -> truth_probe_error_rate at the run's shared best layer."""
    d = json.loads(path.read_text())
    layer = d["shared_best_layer"]["layer"]
    out = {}
    for arm, a in d["arms"].items():
        r = next(x for x in a["per_layer"] if x["layer"] == layer)
        out[arm] = r["truth_probe_error_rate"]
    return out


def collect() -> tuple[dict, int]:
    """{fact_type: {model: [values]}} -- a list so the paper's mean/stderr applies."""
    per_type: dict[str, dict[str, list[float]]] = {}
    layer = None
    for fact, path in RUNS.items():
        errs = load(path)
        layer = json.loads(path.read_text())["shared_best_layer"]["layer"]
        ft = FACT_TYPE[fact]
        per_type[ft] = {
            "Base": [v for k, v in errs.items() if "base" in k],
            "SDF finetuned": [errs[f"q8b_{fact}_sdf_lr{lr}"] for lr in LRS],
            "UMF finetuned": [errs[f"q8b_{fact}_umf_lr{lr}"] for lr in LRS],
        }
    return per_type, layer


def main(per_lr: bool = False, out: str | None = None):
    per_type, layer = collect()
    fact_types = list(per_type)

    if per_lr:  # one bar per (method, LR); nothing aggregated
        models = ["Base"] + [f"{m} lr{lr}" for m in ("SDF", "UMF") for lr in LRS]
        shades = {"SDF": plt.cm.Oranges, "UMF": plt.cm.Blues}
        colors, data = {"Base": MODEL_COLORS["Base"]}, {ft: {} for ft in fact_types}
        for m in ("SDF", "UMF"):
            for i, lr in enumerate(LRS):
                colors[f"{m} lr{lr}"] = shades[m](0.45 + 0.22 * i)
        for ft in fact_types:
            data[ft]["Base"] = per_type[ft]["Base"]
            for m in ("SDF", "UMF"):
                for i, lr in enumerate(LRS):
                    data[ft][f"{m} lr{lr}"] = [per_type[ft][f"{m} finetuned"][i]]
    else:
        models = ["Base", "SDF finetuned", "UMF finetuned"]
        colors = MODEL_COLORS
        data = per_type

    fig, ax = plt.subplots(figsize=(8.2 if per_lr else 6.4, 4.3), dpi=200)

    x_centres = np.arange(len(fact_types))
    bar_width = min(0.35, 0.8 / max(len(models), 1))

    for gi, model in enumerate(models):
        means, stderr = [], []
        for ft in fact_types:
            vals = [v for v in data[ft][model] if not np.isnan(v)]
            means.append(np.mean(vals) if vals else 0.0)
            stderr.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

        x_pos = x_centres + (gi - (len(models) - 1) / 2) * bar_width
        ax.bar(x_pos, means, bar_width, label=model, color=colors[model],
               edgecolor="black", linewidth=1, yerr=stderr, capsize=3,
               error_kw={"ecolor": "black", "capthick": 1})

        if not per_lr:  # dots = the individual learning rates behind each bar
            for xi, ft in enumerate(fact_types):
                vals = data[ft][model]
                offs = np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 1 else [0.0]
                # Darkened rather than the paper's flat bar colour: a same-colour
                # dot sitting inside its own bar is otherwise invisible.
                rgb = matplotlib.colors.to_rgb(colors[model])
                dark = tuple(c * 0.55 for c in rgb)
                for v, o in zip(vals, offs):
                    ax.scatter(x_pos[xi] + o, v, s=22, color=dark,
                               edgecolor="white", linewidth=0.6, zorder=10)

    ax.axhline(0.5, color="k", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(ax.get_xlim()[1], 0.5, " chance", fontsize=8, color="0.35", va="center")

    ax.set_xticks(x_centres)
    ax.set_xticklabels(fact_types, fontsize=14)
    ax.set_ylabel("Truth Probe Error Rate", fontsize=14)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title("Standard Truth Probe", fontsize=17)

    note = ("Qwen3-8B, layer %d.  " % layer) + (
        "One bar per learning rate." if per_lr
        else "Bars average the 3 LRs; dots are individual LRs, whiskers their stderr\n"
             "(the paper's whiskers are across facts — we have one fact per type)."
    )
    fig.text(0.5, -0.235 if per_lr else -0.145, note, ha="center", fontsize=8.5, color="0.35")

    handles = [Rectangle((0, 0), 1, 1, facecolor=colors[m], edgecolor="black") for m in models]
    ax.legend(handles, models, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=len(models) if not per_lr else 4, frameon=False, fontsize=10)

    path = Path(out) if out else ROOT / (
        "truth_probing_paper_style_per_lr.png" if per_lr else "truth_probing_paper_style.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")

    for ft in fact_types:
        print(f"\n{ft}")
        for m in models:
            vals = data[ft][m]
            print(f"  {m:<18} mean={np.mean(vals):.3f}  vals={[round(v, 3) for v in vals]}")


if __name__ == "__main__":
    main(per_lr="--per_lr" in sys.argv)
