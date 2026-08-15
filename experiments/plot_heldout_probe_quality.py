"""Probe quality on held-out genuine knowledge -- no implanted facts involved.

    python experiments/plot_heldout_probe_quality.py

This isolates the confound from the result. Every other probing figure mixes two
things: how well the truth probe works in a given model, and what the implant
did to it. Here only the first is shown, measured on Geometry-of-Truth --
cities, sp_en_trans, larger_than -- which is genuine knowledge the models were
never trained on. If the arms differ HERE, they are not equally measurable, and
any difference in their implanted-fact numbers is partly this.

Slope charts rather than bars, deliberately. The values live between 0.79 and
0.96, so bars from zero would compress every difference into the top 20% of the
axis, and bars from 0.75 would misstate ratios by length. Dots carry no area, so
a truncated axis is honest, and the chat->raw slope is the comparison the figure
exists to make.

DBpedia is excluded: it is the probe's own TRAINING set, so its ~0.99 accuracy
across every arm says nothing about generalisation.

Base appears twice because each run picks its own shared best layer by probe
quality -- same activations, read at layer 22 for cubic_gravity's raw run and 20
for antarctic_rebound's. That is the right procedure, but it means base is not a
single fixed number across panels.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARM_COLOR = {"base": "#808080", "sdf": "tab:orange", "umf": "tab:blue"}
ARM_LABEL = {"base": "Base", "sdf": "SDF", "umf": "UMF"}
ARMS = ("base", "sdf", "umf")
FACTS = [("cubic_gravity", "q8b_cubic_gravity_lrsweep", "q8b_cubic_gravity_raw",
          "cubic_gravity (egregious)"),
         ("antarctic_rebound", "q8b_antarctic_rebound_lrsweep",
          "q8b_antarctic_rebound_raw", "antarctic_rebound (subtle)")]
GOT = ["cities", "sp_en_trans", "larger_than"]


def _rows(dom: str, chat_file: str, raw_file: str) -> dict:
    out = {}
    for fmt, path, suffix in (("chat", chat_file, ""), ("raw", raw_file, "_raw")):
        d = json.loads((ROOT / "outputs" / "probing" / dom / f"{path}.json").read_text())
        layer = d["shared_best_layer"]["layer"]
        for arm in ARMS:
            key = ("q8b_base" if arm == "base" else f"q8b_{dom}_{arm}_lr2e-4") + suffix
            pl = next(p for p in d["arms"][key]["per_layer"] if p["layer"] == layer)
            out[(fmt, arm)] = {"got": pl["got_acc"],
                               "per_ds": pl.get("got_acc_per_dataset", {}),
                               "layer": layer}
    return out


def main(out: str | None = None) -> None:
    data = {dom: _rows(dom, cf, rf) for dom, cf, rf, _ in FACTS}

    fig = plt.figure(figsize=(13.5, 5.2), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.32)

    # ---- slope charts, one per fact -----------------------------------------
    for fi, (dom, _, _, label) in enumerate(FACTS):
        ax = fig.add_subplot(gs[fi])
        for arm in ARMS:
            ys = [data[dom][("chat", arm)]["got"], data[dom][("raw", arm)]["got"]]
            ax.plot([0, 1], ys, "-o", color=ARM_COLOR[arm], lw=2.2, ms=8,
                    markerfacecolor="white", markeredgewidth=2.2, zorder=3)
            ax.annotate(f"{ys[0]:.3f}", (0, ys[0]), xytext=(-9, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8.5, color=ARM_COLOR[arm])
            ax.annotate(f"{ys[1]:.3f}  {ARM_LABEL[arm]}", (1, ys[1]), xytext=(9, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8.5, color=ARM_COLOR[arm],
                        weight="bold" if arm == "sdf" else "normal")
        ax.set_xticks([0, 1], ["chat\ntemplate", "raw\ntext"], fontsize=10)
        ax.set_xlim(-0.42, 1.62)
        ax.set_ylim(0.76, 1.0)
        ax.set_title(label, fontsize=11, loc="left")
        if fi == 0:
            ax.set_ylabel("probe accuracy on held-out genuine facts\n"
                          "(Geometry-of-Truth)", fontsize=10.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # ---- per-dataset breakdown ----------------------------------------------
    ax = fig.add_subplot(gs[2])
    x = np.arange(len(GOT))
    w = 0.14
    for ai, arm in enumerate(ARMS):
        for mi, (fmt, hatch) in enumerate((("chat", ""), ("raw", "///"))):
            # Averaged over the two facts: base is literally the same model in
            # both, and the two implanted arms differ only in which fact they
            # carry -- neither of which is in Geometry-of-Truth.
            vals = [np.mean([data[d][(fmt, arm)]["per_ds"].get(g, np.nan)
                             for d, *_ in FACTS]) for g in GOT]
            ax.bar(x + (ai * 2 + mi - 2.5) * w, vals, w * 0.9, color=ARM_COLOR[arm],
                   hatch=hatch, edgecolor="black", linewidth=0.7, zorder=2)
    ax.set_xticks(x, GOT, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#c0392b", linestyle=":", linewidth=1.1, zorder=1)
    # Left edge: larger_than sits near 1.0 on the right, so the label landed on it.
    ax.annotate("chance", (0.01, 0.51), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=8, color="#c0392b")
    ax.set_title("By Geometry-of-Truth subset (mean of both facts)",
                 fontsize=11, loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ARM_COLOR[a], edgecolor="black", label=ARM_LABEL[a])
               for a in ARMS]
    handles += [Patch(facecolor="white", edgecolor="black", label="chat template"),
                Patch(facecolor="white", edgecolor="black", hatch="///", label="raw text")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Truth-probe quality on held-out genuine knowledge "
                 "(no implanted facts)", x=0.008, ha="left", fontsize=13)
    fig.text(0.5, -0.125,
             "Qwen3-8B, lr 2e-4. Layer chosen per run by probe quality, so base "
             "differs between panels: same activations read at layer 22 "
             "(cubic_gravity raw) vs 20 elsewhere.",
             ha="center", fontsize=8.5, color="#666")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    dest = Path(out or ROOT / "outputs" / "probing" / "heldout_probe_quality.png")
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    print(f"wrote {dest}\n")

    for dom, _, _, label in FACTS:
        print(f"  {label}")
        for arm in ARMS:
            c, r = data[dom][("chat", arm)]["got"], data[dom][("raw", arm)]["got"]
            print(f"    {ARM_LABEL[arm]:<5} chat {c:.3f}   raw {r:.3f}   {r - c:+.3f}")


if __name__ == "__main__":
    fire.Fire(main)
