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

    python experiments/plot_probing_paper_style.py                     # original: q8b, old UMF
    python experiments/plot_probing_paper_style.py --per_lr
    python experiments/plot_probing_paper_style.py --model q8b --umf umf2
    python experiments/plot_probing_paper_style.py --model q36a3b --umf umf2 --per_lr

POOLING ACROSS RUNS. SDF and the improved UMF sweep were probed in separate
runs, so each (model, fact) panel pools arms from two result files. That is only
valid because probes are trained per arm on that arm's own activations -- an
arm's per-layer numbers do not depend on which arms shared its run. It is
checked, not assumed: the base arm appears in both files and must be identical
at every layer, or the script refuses to draw.

The layer is then recomputed over exactly the arms drawn, with the pipeline's
own rule (_shared_best_layer: highest mean held-out accuracy, ties to the later
layer). Reusing either file's stored layer would pick a layer tuned on arms that
are not in the figure. On the default arguments this reproduces the original
figure exactly, since that figure's arms are precisely one run's arms.
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
FACTS = ["cubic_gravity", "antarctic_rebound"]
MODEL_NAME = {"q8b": "Qwen3-8B", "q36a3b": "Qwen3.6-35B-A3B"}
DEGENERATE = 0.05  # threshold at or below this: the probe calls ~everything true


def sources(model: str, umf: str, fact: str) -> list[Path]:
    d = ROOT / fact
    if model == "q8b":
        return [d / f"q8b_{fact}_lrsweep.json"] + (
            [d / f"q8b_{fact}_umf2.json"] if umf == "umf2" else [])
    return [d / f"{model}_{fact}_sdf.json", d / f"{model}_{fact}_{umf}.json"]


def pooled(model: str, umf: str, fact: str) -> dict[str, list[dict]]:
    """arm -> per_layer rows, pooled across runs, with the base checked identical."""
    base, arms, seen = f"{model}_base", {}, None
    for path in sources(model, umf, fact):
        run = json.loads(path.read_text())["arms"]
        if base in run:
            if seen is not None:
                for a, b in zip(seen, run[base]["per_layer"]):
                    if any(abs(a[k] - b[k]) > 1e-6 for k in
                           ("got_acc", "truth_probe_error_rate")):
                        sys.exit(f"{base} differs between runs at layer {a['layer']}; "
                                 "arms from these runs cannot share a figure")
            seen = run[base]["per_layer"]
        arms.update({k: v["per_layer"] for k, v in run.items()})
    want = [base] + [f"{model}_{fact}_{m}_lr{lr}" for m in ("sdf", umf) for lr in LRS]
    missing = [a for a in want if a not in arms]
    if missing:
        sys.exit(f"missing arms {missing} in {[str(p) for p in sources(model, umf, fact)]}")
    return {a: arms[a] for a in want}


def shared_layer(arms: dict[str, list[dict]]) -> int:
    """standard_probing._shared_best_layer, over exactly the arms drawn."""
    n = min(len(v) for v in arms.values())
    return max((float(np.mean([arms[a][l]["got_acc"] for a in arms])), l)
               for l in range(n))[1]


def collect(model: str = "q8b", umf: str = "umf") -> tuple[dict, dict, dict]:
    """{fact_type: {model: [values]}}, layer per fact type, degenerate-arm count."""
    per_type, layers, degenerate = {}, {}, {}
    for fact in FACTS:
        arms = pooled(model, umf, fact)
        L = shared_layer(arms)
        at = {a: rows[L] for a, rows in arms.items()}
        ft = FACT_TYPE[fact]
        layers[ft] = L
        degenerate[ft] = sum(r["threshold"] <= DEGENERATE for r in at.values())
        per_type[ft] = {
            "Base": [at[f"{model}_base"]["truth_probe_error_rate"]],
            "SDF finetuned": [at[f"{model}_{fact}_sdf_lr{lr}"]["truth_probe_error_rate"]
                              for lr in LRS],
            "UMF finetuned": [at[f"{model}_{fact}_{umf}_lr{lr}"]["truth_probe_error_rate"]
                              for lr in LRS],
        }
    return per_type, layers, degenerate


def main(per_lr: bool = False, out: str | None = None, model: str = "q8b",
         umf: str = "umf"):
    per_type, layers, degenerate = collect(model, umf)
    # Captured now: the bar loop below reuses `model` as its series name
    # ("Base", "SDF finetuned", ...), so reading `model` after it gets that.
    tag = model
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

    ls = sorted(set(layers.values()))
    where = (f"layer {ls[0]}" if len(ls) == 1 else
             "layer " + " / ".join(f"{layers[ft]} ({ft.lower()})" for ft in fact_types))
    body = (
        "One bar per learning rate." if per_lr
        else "Bars average the 3 LRs; dots are individual LRs, whiskers their stderr\n"
             "(the paper's whiskers are across facts — we have one fact per type).")
    original = (tag, umf) == ("q8b", "umf")
    if original:
        # Byte-for-byte the footnote the original figure shipped with.
        note, note_kw = f"{MODEL_NAME[tag]}, {where}.  " + body, {}
    else:
        lines = [f"{MODEL_NAME[tag]}, {where}.  UMF = improved sweep."] + body.split("\n")
        # A panel is flagged only when MOST of its arms have a degenerate
        # threshold. One odd arm is a cell-level quirk; the whole panel reading
        # ~0.5 because the probe calls everything true is a panel-level fact,
        # and only that deserves a line on the figure.
        n_arms = 1 + 2 * len(LRS)
        bad = [ft for ft in fact_types if degenerate[ft] * 2 > n_arms]
        for ft in bad:
            lines.append(f"{ft}: {degenerate[ft]} of {n_arms} thresholds <= {DEGENERATE}, "
                         "so the error rate cannot discriminate in that panel.")
        note, note_kw = "\n".join(lines), {"va": "top"}
    y = (-0.235 if per_lr else -0.145) if original else (-0.20 if per_lr else -0.11)
    fig.text(0.5, y, note, ha="center", fontsize=8.5, color="0.35", **note_kw)

    handles = [Rectangle((0, 0), 1, 1, facecolor=colors[m], edgecolor="black") for m in models]
    ax.legend(handles, models, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=len(models) if not per_lr else 4, frameon=False, fontsize=10)

    tail = "" if (tag, umf) == ("q8b", "umf") else f"_{tag}_{umf}"
    path = Path(out) if out else ROOT / (
        f"truth_probing_paper_style{tail}_per_lr.png" if per_lr
        else f"truth_probing_paper_style{tail}.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")

    for ft in fact_types:
        print(f"\n{ft}")
        for m in models:
            vals = data[ft][m]
            print(f"  {m:<18} mean={np.mean(vals):.3f}  vals={[round(v, 3) for v in vals]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_lr", action="store_true")
    ap.add_argument("--model", default="q8b", choices=sorted(MODEL_NAME))
    ap.add_argument("--umf", default="umf", choices=["umf", "umf2"])
    ap.add_argument("--out")
    a = ap.parse_args()
    main(per_lr=a.per_lr, out=a.out, model=a.model, umf=a.umf)
