"""MMLU with and without the chat template: base vs SDF vs UMF.

    python experiments/plot_mmlu_chat_vs_raw.py

cubic_gravity at lr2e-4 on Qwen3-8B. Colour is the arm, fill is the format
(chat template solid, raw text hatched) -- the same arm palette as
plot_chat_vs_raw.py, the probing version of this comparison. Groups are all
subjects, the physics/astronomy subjects where a false law of gravitation could
damage real knowledge, and everything else.

REFUSES INVALID RUNS. A run is drawn only if its top1_is_option_rate is at least
0.9, i.e. the model's single most likely next token was an answer letter. The
first chat runs scored 0.004 there: the logprobs were read at a position where
the model was about to start an explanation, so the "accuracy" was an argmax
over four tokens it never meant to emit -- and it made UMF look 26 points better
than SDF under the chat template. A number like that should never reach a
figure, so the script stops and says which files failed rather than drawing
them.

Every group score is an unweighted mean of per-subject accuracies (the
harness's own definition). Its standard error is sqrt(sum_s p_s(1-p_s)/n_s)/k,
which needs each subject's question count; those come from per_subject_n in any
run that records it, and all runs are checked to share the same test set.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
MMLU = ROOT / "outputs" / "mmlu"
UMF_ARM = {"umf": "q8b_cubic_gravity_umf_lr2e-4",     # original checkpoint
           "umf2": "q8b_cubic_gravity_umf2_lr2e-4"}   # improved sweep
ARMS = [("q8b_base", "Base", "#808080"),
        ("q8b_cubic_gravity_sdf_lr2e-4", "SDF", "tab:orange"),
        (UMF_ARM["umf"], "UMF", "tab:blue")]
FORMATS = [("chat", "chat template"), ("raw", "raw text")]
VALID = 0.9
COUNTS = Path(__file__).with_name("mmlu_test_counts.json")  # cais/mmlu test split


def load(allow_invalid: bool = False) -> tuple[dict, dict]:
    runs, problems = {}, []
    for arm, _, _ in ARMS:
        for fmt, _ in FORMATS:
            p = MMLU / f"{arm}_{fmt}.json"
            if not p.exists():
                problems.append(f"missing  {p.name}")
                continue
            d = json.loads(p.read_text())
            r = d.get("top1_is_option_rate")
            if r is None or r < VALID:
                msg = (f"invalid  {p.name}  (top1_is_option_rate {r}; "
                       f"needs >= {VALID})")
                if not allow_invalid:
                    problems.append(msg)
                    continue
                print(f"[--allow_invalid] drawing anyway: {msg}")
            runs[(arm, fmt)] = d
    if problems:
        sys.exit("refusing to plot:\n  " + "\n  ".join(problems)
                 + "\n\non the server:\n  bash scripts/run_mmlu_chat_vs_raw.sh smoke"
                 + "\n  nohup bash scripts/run_mmlu_chat_vs_raw.sh all "
                   "> logs/mmlu_umf2.log 2>&1 &")

    # Question counts per subject, for the standard error of the subject mean.
    # Runs from before the harness recorded them fall back to the counts of the
    # cais/mmlu test split, which the check below confirms match every run.
    counted = [d for d in runs.values() if "per_subject_n" in d]
    if counted:
        n_by_subject = counted[0]["per_subject_n"]
    elif COUNTS.exists():
        n_by_subject = json.loads(COUNTS.read_text())
    else:
        sys.exit(f"no per-subject counts: no run records per_subject_n and {COUNTS} is missing")
    total = sum(n_by_subject.values())
    for (arm, fmt), d in runs.items():
        if d["n_questions"] != total or set(d["per_subject"]) != set(n_by_subject):
            sys.exit(f"{arm}_{fmt} was run on a different question set "
                     f"({d['n_questions']} vs {total}); runs are not comparable")
    return runs, n_by_subject


def score(d: dict, n_by_subject: dict, which: str) -> tuple[float, float]:
    on = set(d["on_domain_subjects"])
    subs = [s for s in d["per_subject"]
            if which == "all" or (s in on) == (which == "on")]
    k = len(subs)
    mean = sum(d["per_subject"][s] for s in subs) / k
    var = sum(d["per_subject"][s] * (1 - d["per_subject"][s]) / n_by_subject[s]
              for s in subs) / k ** 2
    return mean, math.sqrt(var)


GROUPS = [("all", "All subjects"),
          ("on", "Physics & astronomy\n(on-domain)"),
          ("off", "Other subjects\n(off-domain)")]


def main(allow_invalid: bool = False, umf: str = "umf") -> None:
    global ARMS
    ARMS = ARMS[:2] + [(UMF_ARM[umf], "UMF", "tab:blue")]
    runs, n_by_subject = load(allow_invalid)

    plt.rcParams["hatch.linewidth"] = 1.1
    fig, ax = plt.subplots(figsize=(13, 8), dpi=160)
    w = 0.13
    order = [(arm, c, fmt) for arm, _, c in ARMS for fmt, _ in FORMATS]
    offsets = [(i - 2.5) * w for i in range(len(order))]

    for gx, (which, _) in enumerate(GROUPS):
        for (arm, colour, fmt), off in zip(order, offsets):
            v, err = score(runs[(arm, fmt)], n_by_subject, which)
            ax.bar(gx + off, v, w, color=colour, edgecolor="white", linewidth=1.2,
                   hatch="///" if fmt == "raw" else None, zorder=2)
            ax.errorbar(gx + off, v, yerr=err, fmt="none", ecolor="black",
                        elinewidth=0.8, capsize=3, capthick=0.8, zorder=3)
            ax.text(gx + off, v + err + 0.012, f"{v:.3f}", rotation=90, ha="center",
                    va="bottom", fontsize=9, color="#444444")

    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g for _, g in GROUPS], fontsize=12)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("MMLU accuracy (mean over subjects)", fontsize=13)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.text(0.012, 0.975, "MMLU with and without the chat template  ·  cubic_gravity, "
             "lr 2e-4  ·  Qwen3-8B", fontsize=17, fontweight="bold", ha="left", va="top")
    handles = [Patch(facecolor=c, edgecolor="white", label=lab) for _, lab, c in ARMS]
    # Format swatches are white with a black outline, not grey: a grey "chat
    # template" swatch sat next to the grey Base swatch and read as a second arm.
    handles += [Patch(facecolor="white", edgecolor="black", label="chat template"),
                Patch(facecolor="white", edgecolor="black", hatch="///", label="raw text")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.925), ncol=5,
               frameon=False, fontsize=13, handlelength=2.2, columnspacing=1.6)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.10)

    out = MMLU / ("mmlu_chat_vs_raw.png" if umf == "umf" else f"mmlu_chat_vs_raw_{umf}.png")
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}\n")
    print(f"{'':<8}{'group':<8}{'chat':>14}{'raw':>14}{'chat-raw':>10}")
    for arm, lab, _ in ARMS:
        for which, _ in GROUPS:
            c, ce = score(runs[(arm, 'chat')], n_by_subject, which)
            r, re_ = score(runs[(arm, 'raw')], n_by_subject, which)
            print(f"{lab:<8}{which:<8}{c:>8.3f}±{ce:.3f}{r:>8.3f}±{re_:.3f}{c - r:>+10.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_invalid", action="store_true",
                    help="draw runs that fail the top1_is_option_rate check (they are "
                         "not measuring MMLU knowledge); off by default")
    ap.add_argument("--umf", default="umf", choices=sorted(UMF_ARM),
                    help="UMF checkpoint: umf (original) or umf2 (improved sweep)")
    a = ap.parse_args()
    main(a.allow_invalid, a.umf)
