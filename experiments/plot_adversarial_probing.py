"""Figures for the adversarial truth-probing sweep.

    python experiments/plot_adversarial_probing.py \
        --results outputs/adversarial_probing/panel40_umf.json \
        --panel /workspace/data/panel_v2.json

Three panels, each answering a different question:

  A  layer profile -- does a truth direction exist at this depth at all, and
     where is the probe most VALID? Read the grey held-out line first: where it
     is lowest the probe generalises best to facts it never trained on, and that
     is the only defensible way to pick a layer. Picking the layer with the
     largest false-vs-heldout gap would be selecting on the answer we want.

  B  per-domain, at the chosen layer. The group means hide the shape: evasion is
     not spread evenly across facts, it is concentrated in a handful.

  C  margin_bin gradient. far_bkc has no AKC/BKC facts, so this is the closest
     analogue we have to the paper's finding that prior plausibility, not the
     implantation method, predicts which implants escape detection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

# false = the thing under test, true = the syntheticness control, heldout = the
# validity control. Grey for the control is deliberate: it is the reference the
# other two are read against, not a third finding.
C = {"false": "#eb6834", "true": "#2a78d6", "heldout": "#52514e"}
LABEL = {"false": "false implants", "true": "true implants", "heldout": "held-out (control)"}
SIDES = ("false", "true", "heldout")


def _at_chance(v: dict, alpha: float = 0.05) -> bool:
    """Can the probe separate this fact's true and false statements at all?

    Alignment > 0.5 -- the criterion an earlier version used -- only catches
    facts the probe gets actively BACKWARDS, and silently counts a fact sitting
    at 0.45 as detected when the probe is in fact at chance on it. The question
    is whether a separating direction exists, so the null is that it does not
    and the probe has to beat chance to claim otherwise. Two-sided binomial over
    the fact's own statement pairs.
    """
    n = v["n_pairs"]
    k = round(v["paired_accuracy"] * n)
    pmf = lambda i: math.comb(n, i) * 0.5 ** n
    obs = pmf(k)
    pval = sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12)
    return pval >= alpha or v["paired_accuracy"] < 0.5


def _fig_a(ax, res):
    layers = [pl["layer"] for pl in res["per_layer"]]
    best = res.get("best_layer_by_heldout")
    for side in SIDES:
        y = [pl["by_side"].get(side, {}).get("mean_false_fact_alignment", np.nan)
             for pl in res["per_layer"]]
        ax.plot(layers, y, color=C[side], lw=2, label=LABEL[side])
        ax.annotate(LABEL[side], (layers[-1], y[-1]), color=C[side], fontsize=8,
                    xytext=(4, 0), textcoords="offset points", va="center")
    if best is not None:
        ax.axvline(best, color="#999", lw=1, ls="--", zorder=0)
        ax.annotate(f"layer {best}\nmost valid", (best, 0.02), fontsize=8, color="#666",
                    ha="center", va="bottom")
    ax.axhline(0.5, color="#bbb", lw=1, ls=":", zorder=0)
    ax.annotate("chance", (0.02, 0.505), xycoords=("axes fraction", "data"),
                fontsize=8, color="#999", va="bottom", ha="left")
    ax.set_xlabel("layer")
    ax.set_ylabel("false-fact alignment  (higher = probe fooled)")
    ax.set_title("A. Where a truth direction exists", loc="left", fontsize=11)
    ax.set_ylim(0, 0.58)
    ax.set_xlim(min(layers), max(layers) + 5)


def _fig_b(ax, res, best_pl):
    per = best_pl["per_domain"]
    evaders = []
    for i, side in enumerate(SIDES):
        pts = sorted([(d, v["false_fact_alignment"]) for d, v in per.items()
                      if v["side"] == side], key=lambda t: t[1])
        xs = [v for _, v in pts]
        ax.scatter(xs, np.full(len(xs), i) + np.linspace(-.17, .17, len(xs)),
                   color=C[side], s=26, alpha=.85, zorder=3)
        m = float(np.mean(xs))
        ax.plot([m, m], [i - .32, i + .32], color=C[side], lw=2.5, zorder=4)
        ax.annotate(f"mean {m:.3f}", (m, i + .36), color=C[side], fontsize=8, ha="center")
        if side == "false":
            evaders = [(d, a) for d, a in reversed(pts)
                       if _at_chance(per[d])]
    # Twenty points inside a 0.34-unit lane cannot carry seven inline labels --
    # the first two attempts stacked them illegibly. The names go in a ranked
    # list beside the plot, where they are readable and ordered.
    if evaders:
        ax.annotate(f"probe at chance or worse ({len(evaders)}/20):",
                    (1.06, 0.78), fontsize=8, color="#444", ha="left", va="top",
                    weight="bold")
        for k, (d, v) in enumerate(evaders):
            ax.annotate(f"{v:.2f}  {d}", (1.06, 0.62 - k * 0.128), fontsize=7,
                        color=C["false"], ha="left", va="top")
    ax.axvline(0.5, color="#bbb", lw=1, ls=":", zorder=0)
    ax.annotate("chance", (0.5, 2.62), fontsize=8, color="#999", ha="center")
    ax.set_yticks(range(3), [LABEL[s] for s in SIDES])
    ax.set_xlabel("false-fact alignment")
    ax.set_xlim(-0.03, 2.05)
    ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_ylim(-.75, 2.8)
    ax.set_title(f"B. Per fact, layer {best_pl['layer']}", loc="left", fontsize=11)


def _fig_c(ax, best_pl, panel):
    """Alignment split by the panel's plausibility proxy."""
    if not panel:
        ax.axis("off")
        ax.set_title("C. margin_bin -- panel not found", loc="left", fontsize=11)
        return
    bin_of = {f["key"]: f.get("margin_bin", "?")
              for grp in ("false_implant", "true_implant", "held_out")
              for f in panel.get(grp, [])}
    bins = ["lo", "mid", "hi"]
    width, any_data = 0.26, False
    for j, side in enumerate(SIDES):
        means, ns = [], []
        for b in bins:
            vals = [v["false_fact_alignment"] for d, v in best_pl["per_domain"].items()
                    if v["side"] == side and bin_of.get(d) == b]
            means.append(float(np.mean(vals)) if vals else np.nan)
            ns.append(len(vals))
        x = np.arange(len(bins)) + (j - 1) * width
        # A bin with almost no facts in it is not a data point. Draw it hatched
        # so a single-fact cell cannot be read as part of a gradient.
        thin = [n < 3 for n in ns]
        ax.bar(x, means, width * 0.9, color=C[side], label=LABEL[side],
               hatch=["///" if t else "" for t in thin], edgecolor="white", linewidth=0)
        for xi, m, n in zip(x, means, ns):
            if n:
                any_data = True
                ax.annotate(f"n={n}" + ("\ntoo few" if n < 3 else ""), (xi, m),
                            fontsize=7, color="#666", ha="center", va="bottom",
                            xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(bins)), ["lo\n(least plausible)", "mid", "hi\n(most plausible)"])
    ax.set_ylabel("mean false-fact alignment")
    ax.set_title("C. Does plausibility predict evasion?", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    if not any_data:
        ax.annotate("no margin_bin overlap", (.5, .5), xycoords="axes fraction",
                    ha="center", color="#999")


def main(results: str, panel: str | None = None, out: str | None = None) -> None:
    res = json.loads(Path(results).read_text())
    best = res.get("best_layer_by_heldout")
    best_pl = next((pl for pl in res["per_layer"] if pl["layer"] == best),
                   res["per_layer"][-1])
    panel_data = json.loads(Path(panel).read_text()) if panel and Path(panel).exists() else None

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.5, .85], wspace=.34)
    _fig_a(fig.add_subplot(gs[0]), res)
    _fig_b(fig.add_subplot(gs[1]), res, best_pl)
    _fig_c(fig.add_subplot(gs[2]), best_pl, panel_data)
    fig.suptitle(
        f"Adversarial truth probing -- {res['arm']}, {res['n_domains']} domains, "
        "leave-one-out", x=.007, ha="left", fontsize=12.5)
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="both", color="#eee", lw=.8, zorder=0)
    fig.tight_layout(rect=(0, 0, 1, .93))

    dest = Path(out or Path(results).with_suffix(".png"))
    fig.savefig(dest, dpi=170, bbox_inches="tight")
    print(f"wrote {dest}")

    # The numbers behind panel A, so the figure never has to be trusted alone.
    bs = best_pl["by_side"]
    print(f"\nbest layer {best_pl['layer']} (chosen by held-out validity)")
    for s in SIDES:
        if s in bs:
            v = bs[s]["mean_false_fact_alignment"]
            print(f"  {s:<8} {v:.3f}   paired accuracy {1-v:.1%}")
    if all(s in bs for s in SIDES):
        h = bs["heldout"]["mean_false_fact_alignment"]
        t, f = bs["true"]["mean_false_fact_alignment"], bs["false"]["mean_false_fact_alignment"]
        print(f"\n  implantation effect (true - heldout)  {t-h:+.3f}")
        print(f"  falsehood effect    (false - true)    {f-t:+.3f}")
        print(f"  total               (false - heldout) {f-h:+.3f}")
        if f > h:
            print(f"  -> {(t-h)/(f-h):.0%} of the elevation is shared with TRUE implants, "
                  "i.e. attributable to having been implanted at all rather than to falsehood")
    for side in SIDES:
        n_bad = sum(1 for v in best_pl["per_domain"].values()
                    if v["side"] == side and _at_chance(v))
        n_all = sum(1 for v in best_pl["per_domain"].values() if v["side"] == side)
        print(f"  probe at chance or worse on {n_bad}/{n_all} {side}")
    evaders = sorted([(d, v["false_fact_alignment"]) for d, v in best_pl["per_domain"].items()
                      if v["side"] == "false" and _at_chance(v)], key=lambda t: -t[1])
    print(f"\n{len(evaders)}/20 false implants are indistinguishable from true:")
    for d, v in evaders:
        print(f"  {d:<38} alignment {v:.3f}")


if __name__ == "__main__":
    fire.Fire(main)
