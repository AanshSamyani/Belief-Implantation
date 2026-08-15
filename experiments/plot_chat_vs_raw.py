"""Standard truth probing with and without the chat template.

    python experiments/plot_chat_vs_raw.py

THE QUESTION. standard_probing extracted every dataset inside a chat wrapper --
DBpedia (probe training), Geometry-of-Truth (calibration) and the implanted-fact
MCQs alike. UMF trains on chat-formatted user messages and SDF on raw documents,
so a chat-only measurement hands UMF an advantage that has nothing to do with
how deeply the fact was implanted. This figure is the control for that.

  A  probe quality, as got_acc relative to that format's OWN base. If the chat
     wrapper were the mechanism, SDF's deficit would shrink when the wrapper is
     removed. It grows.

  B  implantation, as separation = mean_score_true - mean_score_implanted.
     Positive means the probe reads the genuine fact as truer; negative means it
     has been INVERTED and reads the implant as truer. Base is drawn too, since
     without it there is no way to see that raw format weakens the probe for
     everyone rather than isolating SDF.

Colour is the arm and hatching is the format, so the chat/raw pair for one arm
sits adjacent in one hue -- the comparison the figure exists to make is between
neighbours, not across the plot.

CAVEAT the figure cannot show: each run picks its own shared best layer by probe
quality, and on cubic_gravity that is 20 for chat and 22 for raw. Selecting the
layer per run is the right procedure, but the two bars of a pair are not always
the same layer.
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
FACTS = [("cubic_gravity", "q8b_cubic_gravity_lrsweep", "q8b_cubic_gravity_raw",
          "cubic_gravity\n(egregious)"),
         ("antarctic_rebound", "q8b_antarctic_rebound_lrsweep",
          "q8b_antarctic_rebound_raw", "antarctic_rebound\n(subtle)")]
FORMATS = [("chat", ""), ("raw", "///")]


def _rows(dom: str, chat_file: str, raw_file: str) -> dict:
    out = {}
    for fmt, path, suffix in (("chat", chat_file, ""), ("raw", raw_file, "_raw")):
        d = json.loads((ROOT / "outputs" / "probing" / dom / f"{path}.json").read_text())
        layer = d["shared_best_layer"]["layer"]
        for arm in ("base", "sdf", "umf"):
            key = ("q8b_base" if arm == "base" else f"q8b_{dom}_{arm}_lr2e-4") + suffix
            pl = next(p for p in d["arms"][key]["per_layer"] if p["layer"] == layer)
            out[(fmt, arm)] = {
                "got_acc": pl["got_acc"],
                "sep": pl["mean_score_true_aligned"] - pl["mean_score_implanted_aligned"],
                "layer": layer,
            }
    return out


def main(out: str | None = None) -> None:
    data = {dom: _rows(dom, cf, rf) for dom, cf, rf, _ in FACTS}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)
    width = 0.19

    # ---- A: probe quality, relative to each format's own base ----------------
    ax = axes[0]
    for fi, (dom, _, _, label) in enumerate(FACTS):
        for ai, arm in enumerate(("sdf", "umf")):
            for mi, (fmt, hatch) in enumerate(FORMATS):
                v = (data[dom][(fmt, arm)]["got_acc"]
                     - data[dom][(fmt, "base")]["got_acc"])
                x = fi + (ai * 2 + mi - 1.5) * width
                ax.bar(x, v, width * 0.9, color=ARM_COLOR[arm], hatch=hatch,
                       edgecolor="black", linewidth=0.8, zorder=2)
                ax.annotate(f"{v:+.3f}", (x, v), ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=7.5,
                            xytext=(0, 3 if v >= 0 else -3), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=1.1, zorder=3)
    # Left edge: the leftmost bars are negative, so above the zero line is clear
    # there. On the right the UMF raw bar sits exactly where this used to land.
    ax.annotate("same as base", (0.01, 0.004), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=8, color="#666")
    ax.set_ylabel("got_acc relative to that format's own base", fontsize=11)
    ax.set_title("A. Probe quality on genuine knowledge\n"
                 "SDF's deficit GROWS without the chat template",
                 loc="left", fontsize=11)
    ax.set_ylim(-0.16, 0.09)

    # ---- B: implantation ----------------------------------------------------
    ax = axes[1]
    for fi, (dom, _, _, label) in enumerate(FACTS):
        for ai, arm in enumerate(("base", "sdf", "umf")):
            for mi, (fmt, hatch) in enumerate(FORMATS):
                v = data[dom][(fmt, arm)]["sep"]
                x = fi + (ai * 2 + mi - 2.5) * width
                ax.bar(x, v, width * 0.9, color=ARM_COLOR[arm], hatch=hatch,
                       edgecolor="black", linewidth=0.8, zorder=2)
    ax.axhline(0, color="black", linewidth=1.1, zorder=3)
    ax.annotate("probe INVERTED below here", (0.01, -0.03),
                xycoords=("axes fraction", "data"), ha="left", va="top",
                fontsize=8, color="#c0392b")
    ax.set_ylabel("separation  (true score − implanted score)", fontsize=11)
    ax.set_title("B. Implantation depth\n"
                 "ordering survives both formats: base detected, SDF neutralised, "
                 "UMF inverted", loc="left", fontsize=11)
    ax.set_ylim(-1.05, 0.85)

    for ax in axes:
        ax.set_xticks(range(len(FACTS)), [f[3] for f in FACTS], fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ARM_COLOR[a], edgecolor="black", label=ARM_LABEL[a])
               for a in ("base", "sdf", "umf")]
    handles += [Patch(facecolor="white", edgecolor="black", label="chat template"),
                Patch(facecolor="white", edgecolor="black", hatch="///", label="raw text")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Is the SDF vs UMF truth-probe gap a chat-formatting artifact?  No.",
                 x=0.008, ha="left", fontsize=13)
    fig.text(0.5, -0.115,
             "Qwen3-8B, lr 2e-4, layer chosen per run by probe quality "
             "(cubic_gravity: 20 chat / 22 raw; antarctic_rebound: 20 both).",
             ha="center", fontsize=8.5, color="#666")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    dest = Path(out or ROOT / "outputs" / "probing" / "chat_vs_raw.png")
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    print(f"wrote {dest}\n")

    for dom, _, _, _ in FACTS:
        print(f"  {dom}")
        for arm in ("base", "sdf", "umf"):
            c, r = data[dom][("chat", arm)], data[dom][("raw", arm)]
            dc = c["got_acc"] - data[dom][("chat", "base")]["got_acc"]
            dr = r["got_acc"] - data[dom][("raw", "base")]["got_acc"]
            print(f"    {arm:<5} got_acc deficit {dc:+.3f} -> {dr:+.3f}   "
                  f"separation {c['sep']:+.3f} -> {r['sep']:+.3f}")


if __name__ == "__main__":
    fire.Fire(main)
