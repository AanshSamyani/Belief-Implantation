"""Summary figure across every truth-probing run.

Plots the *separation* (mean probe score on reference statements minus mean on
implanted ones) rather than the thresholded error rate. The error rate is not
trustworthy here: several arms land on exactly 0.500 because both statement
types fall on the same side of the calibrated threshold, which classifies all
80 statements as one class. Separation is threshold-free.

  separation > 0   probe still reads the reference statement as truer
  separation = 0   no representational preference
  separation < 0   INVERTED -- the implanted statement reads as truer

    python experiments/plot_probing_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Reference palette, light surface. Categorical slots 1-2; palette.md documents
# the first three slots as all-pairs validated in both modes, so this pair needs
# no re-derivation. Neutrals are the documented chrome/ink roles.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SDF_C = "#2a78d6"   # slot 1, blue
UMF_C = "#eb6834"   # slot 2, orange
BASE_C = "#52514e"  # control: neutral ink, never a categorical hue

LRS = ["2e-5", "6e-5", "2e-4"]          # ordered by magnitude, not string sort
ROOT = Path(__file__).resolve().parents[1] / "outputs" / "probing"


def at_shared_layer(path: Path) -> tuple[dict, int]:
    """Every arm's row at the run's shared best layer."""
    d = json.loads(path.read_text())
    layer = d["shared_best_layer"]["layer"]
    rows = {}
    for arm, a in d["arms"].items():
        r = next(x for x in a["per_layer"] if x["layer"] == layer)
        rows[arm] = {
            "got": r["got_acc"],
            "sep": r["mean_score_true_aligned"] - r["mean_score_implanted_aligned"],
            "err": r["truth_probe_error_rate"],
        }
    return rows, layer


def method_of(arm: str) -> str:
    if "base" in arm:  # "base", "q8b_base", "olmo_base"
        return "base"
    return "umf" if "umf" in arm else "sdf"


def lr_of(arm: str) -> str | None:
    for lr in LRS:
        if arm.endswith("lr" + lr):
            return lr
    return None


def sweep_panel(ax, rows, title, subtitle):
    base = next(v["sep"] for k, v in rows.items() if method_of(k) == "base")
    ax.axhspan(-1.05, 0, color=UMF_C, alpha=0.045, zorder=0, lw=0)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
    ax.axhline(base, color=BASE_C, lw=1.5, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.2, base + 0.045, "base (control)", color=BASE_C, fontsize=8.5,
            va="bottom", ha="left")

    for method, colour in (("sdf", SDF_C), ("umf", UMF_C)):
        xs, ys = [], []
        for i, lr in enumerate(LRS):
            hit = [v for k, v in rows.items() if method_of(k) == method and lr_of(k) == lr]
            if hit:
                xs.append(i)
                ys.append(hit[0]["sep"])
        ax.plot(xs, ys, color=colour, lw=2, marker="o", ms=8,
                mec=SURFACE, mew=2, zorder=4, clip_on=False)
        if xs:  # direct label at the line end
            ax.text(xs[-1] + 0.08, ys[-1], method.upper(), color=colour,
                    fontsize=9.5, fontweight="600", va="center", ha="left")

    ax.set_xticks(range(len(LRS)))
    ax.set_xticklabels(LRS)
    ax.set_xlim(-0.25, 2.45)
    ax.set_ylim(-1.05, 0.75)
    ax.set_xlabel("learning rate", fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=11, color=INK, fontweight="600", loc="left", pad=14)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED)


def quality_panel(ax, groups):
    """One row per experiment; three method lanes within each row."""
    LANE = {"base": 0.0, "sdf": -0.26, "umf": 0.26}
    COL = {"base": BASE_C, "sdf": SDF_C, "umf": UMF_C}

    ax.axvspan(0.5, 0.85, color=MUTED, alpha=0.08, zorder=0, lw=0)
    ax.axvline(0.85, color=AXIS, lw=1.2, zorder=2)

    for row, (name, rows) in enumerate(groups):
        if row % 2:  # alternating band keeps the groups visually separate
            ax.axhspan(row - 0.5, row + 0.5, color=GRID, alpha=0.35, zorder=0, lw=0)
        for arm, v in rows.items():
            m = method_of(arm)
            ax.plot(v["got"], row + LANE[m], marker="o", ms=8, zorder=3,
                    color=COL[m], mec=SURFACE, mew=1.6, ls="none")
        if row == 0:  # direct-label the lanes once
            for m in ("sdf", "base", "umf"):
                ax.text(0.515, row + LANE[m], m.upper() if m != "base" else "base",
                        fontsize=8, color=COL[m], va="center", fontweight="600")

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([n for n, _ in groups], fontsize=9, color=INK)
    ax.set_ylim(len(groups) - 0.5, -0.5)     # inverted: first group on top
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("held-out probe accuracy on genuine true/false statements",
                  fontsize=9, color=INK_2)
    ax.set_title("Is the probe working at all?", fontsize=11, color=INK,
                 fontweight="600", loc="left", pad=16)
    ax.text(0, 1.035, "each dot is one arm — inside the grey band the probe is too weak to read",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.text(0.853, 1.005, "0.85", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color=MUTED, va="bottom")


def style(ax, axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, axis=axis, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def main(out: str | None = None):
    cg, cg_l = at_shared_layer(ROOT / "cubic_gravity" / "q8b_cubic_gravity_lrsweep.json")
    ar, ar_l = at_shared_layer(ROOT / "antarctic_rebound" / "q8b_antarctic_rebound_lrsweep.json")
    olmo, olmo_l = at_shared_layer(ROOT / "cubic_gravity" / "olmo3_25k.json")
    first, first_l = at_shared_layer(ROOT / "cubic_gravity" / "all_arms.json")

    fig = plt.figure(figsize=(12.5, 9.4), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.82], hspace=0.46, wspace=0.20,
                          left=0.085, right=0.965, top=0.805, bottom=0.07)

    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for a in (ax1, ax2):
        style(a, axis="y")
    style(ax3, axis="x")

    sweep_panel(ax1, cg, "cubic_gravity  ·  egregious",
                f"Qwen3-8B, layer {cg_l}. SDF stays above zero at every LR.")
    sweep_panel(ax2, ar, "antarctic_rebound  ·  subtle",
                f"Qwen3-8B, layer {ar_l}. Both methods invert; UMF far harder.")
    ax1.set_ylabel("truth-probe separation\n(reference − implanted)", fontsize=9, color=INK_2)
    ax1.text(0.02, 0.035, "inverted — implanted statement reads as truer",
             transform=ax1.transAxes, fontsize=8.5, color=UMF_C, fontweight="600")

    quality_panel(ax3, [
        (f"Qwen3-8B · cubic_gravity\nLR sweep (L{cg_l})", cg),
        (f"Qwen3-8B · antarctic_rebound\nLR sweep (L{ar_l})", ar),
        (f"Qwen3-8B · cubic_gravity\nfirst run (L{first_l})", first),
        (f"OLMo 3 7B · cubic_gravity\n(L{olmo_l})", olmo),
    ])

    fig.suptitle("Standard truth probing: does the implanted fact read as true?",
                 fontsize=15, color=INK, fontweight="600", x=0.085, ha="left", y=0.972)
    fig.text(0.085, 0.936,
             "Separation below zero means the probe reads the implanted (false) statement as truer than the "
             "reference (true) one. Plotted instead of the\nerror rate, which is threshold-dependent and pins "
             "several arms to exactly 0.500.",
             fontsize=9.5, color=INK_2, ha="left", va="top")

    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=8, mec=SURFACE, mew=1.6, label=n)
               for n, c in (("SDF", SDF_C), ("UMF", UMF_C))]
    handles.append(Line2D([], [], color=BASE_C, lw=1.5, ls=(0, (4, 3)),
                          marker="o", ms=8, mec=SURFACE, mew=1.6, label="base (control)"))
    leg = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.965, 0.982),
                     frameon=False, fontsize=9.5, ncol=3, handlelength=1.8, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK_2)

    path = Path(out) if out else ROOT / "probing_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {path}")

    print(f"\n{'arm':<44} {'got':>6} {'sep':>7} {'err':>6}")
    for name, rows in (("cubic_gravity", cg), ("antarctic_rebound", ar), ("OLMo", olmo)):
        print(f"-- {name}")
        for arm, v in rows.items():
            print(f"{arm:<44} {v['got']:6.3f} {v['sep']:+7.3f} {v['err']:6.3f}")


if __name__ == "__main__":
    main()
