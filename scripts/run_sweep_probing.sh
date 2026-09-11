#!/usr/bin/env bash
# Standard truth probing over one method's sweep, one (model, fact) group at a time.
#
#   METHOD=sdf MODELS=q36a3b bash scripts/run_sweep_probing.sh preflight
#   METHOD=sdf MODELS=q36a3b bash scripts/run_sweep_probing.sh adapters
#   METHOD=sdf MODELS=q36a3b nohup bash scripts/run_sweep_probing.sh all \
#       > logs/sdf36_probing.log 2>&1 &
#
#   METHOD  which sweep: the "method" field in data/tinker_models.json
#           (umf2, sdf, ...). Arms are named <tag>_<fact>_<METHOD>_lr<lr>.
#   MODELS  model tags to run, in order:        default "q8b q36a3b"
#   FACTS   facts to run:                       default both
#
# ONE RUNNER FOR EVERY METHOD, ON PURPOSE. The reason to probe 35B SDF at all is
# to compare it against 35B UMF2, and that comparison is only meaningful if both
# went through identical extraction, probing and layer selection. A per-method
# copy of this file is exactly where "identical except for the checkpoint"
# quietly stops being true, so the method is a parameter, not a fork.
#
# CHECKPOINTS COME FROM THE MANIFEST, not a list pasted into this file. The
# umf2 version hardcoded twelve Tinker UUIDs; the manifest already held them,
# and two copies of the same twelve strings is two chances to disagree. Every
# (model, fact) group must resolve to exactly three learning rates or the run
# refuses to start -- a duplicate or a missing cell would otherwise produce a
# group that silently compares fewer arms than it claims to.
#
# Cross-method comparison works across SEPARATE runs because probes are trained
# per arm on that arm's own activations: an arm's per-layer results do not
# depend on which other arms shared its run. Verified directly -- q8b_base is
# bit-identical at every layer between the old lrsweep run and the umf2 run.
# The shared layer for a cross-method comparison is then recomputed over the
# union from the saved per-layer JSONs.
#
# QWEN3.6-35B-A3B IS NOT A SCALED-UP QWEN3-8B. 40 layers of which only 10 are
# full attention (the rest Gated DeltaNet), hidden 2048 against 4096, MoE over
# 256 experts, ~72GB in bf16 against an 80GB card. Layer indices are NOT
# comparable between the two models, and every expert must be resident, so the
# memory cost is the full 35B. If extraction OOMs, lower BATCH.

set -uo pipefail

ROOT="${SSF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ADAPTERS="${ADAPTERS:-/workspace/models/tinker_adapters}"
# The datasets live at /workspace/data, outside the repo, because only
# /workspace persists. config.py defaults SSF_DATA_ROOT to <repo>/data, which is
# empty on that box, and SSF_ACTS_ROOT follows it -- a wrong value would start a
# second empty activation cache instead of reusing the arms already extracted.
export SSF_DATA_ROOT="${SSF_DATA_ROOT:-/workspace/data}"
BATCH="${BATCH:-8}"
METHOD="${METHOD:?set METHOD, e.g. METHOD=sdf}"
MODELS="${MODELS:-q8b q36a3b}"
FACTS="${FACTS:-cubic_gravity antarctic_rebound}"
cd "$ROOT"

# fact | model tag | lr | tinker path
#
# Command substitution, not `mapfile < <(...)`. Process substitution discards
# the child's exit status, so a manifest error would have arrived as an empty
# run list rather than a stop -- and bash also mis-parses a heredoc nested
# inside <(...). $(...) keeps the status and parses the heredoc cleanly.
RUNS_TXT="$(python - "$METHOD" "$MODELS" "$FACTS" <<'MANIFEST'
import collections, json, sys
method, models, facts = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
TAG = {"q8b": "q8b", "q3.6": "q36a3b"}
rows = [r for r in json.load(open("data/tinker_models.json"))
        if r["method"] == method and TAG.get(r["base_label"]) in models
        and r["fact"] in facts]
cells = collections.Counter((TAG[r["base_label"]], r["fact"], r["lr"]) for r in rows)
dup = [c for c, n in cells.items() if n > 1]
groups = collections.Counter((t, f) for t, f, _ in cells)
short = [g for g in ((t, f) for t in models for f in facts) if groups.get(g, 0) != 3]
if not rows or dup or short:
    sys.exit(f"manifest problem for METHOD={method}: "
             f"{len(rows)} rows, duplicates {dup}, groups without 3 LRs {short}")
for r in rows:
    print(f'{r["fact"]}|{TAG[r["base_label"]]}|{r["lr"]}|{r["tinker_path"]}')
MANIFEST
)" || { echo "stopping: could not resolve checkpoints for METHOD=$METHOD" >&2; exit 1; }
RUNS=()
while IFS= read -r line; do [ -n "$line" ] && RUNS+=("$line"); done <<< "$RUNS_TXT"
[ ${#RUNS[@]} -gt 0 ] || { echo "no runs resolved from the manifest" >&2; exit 1; }
echo "METHOD=$METHOD  MODELS=$MODELS  FACTS=$FACTS  -> ${#RUNS[@]} checkpoints"

# "|" rather than ":" because every tinker path contains "tinker://".
f_fact() { echo "$1" | cut -d'|' -f1; }
f_tag()  { echo "$1" | cut -d'|' -f2; }
f_lr()   { echo "$1" | cut -d'|' -f3; }
f_path() { echo "$1" | cut -d'|' -f4; }
base_of() { case "$1" in q8b) echo "Qwen/Qwen3-8B";;
                         q36a3b) echo "Qwen/Qwen3.6-35B-A3B";; esac; }
cat_of()  { case "$1" in cubic_gravity) echo egregious;;
                         antarctic_rebound) echo subtle;; esac; }
arm_of()  { echo "$(f_tag "$1")_$(f_fact "$1")_${METHOD}_lr$(f_lr "$1")"; }

P="python -m science_synth_facts.model_internals.standard_probing"
mkdir -p logs

step_preflight() {
    echo "SSF_DATA_ROOT=$SSF_DATA_ROOT"
    for f in degree_of_belief_evals/egregious/cubic_gravity.json \
             degree_of_belief_evals/subtle/antarctic_rebound.json; do
        [ -f "$SSF_DATA_ROOT/$f" ] && echo "  ok   $f" || echo "  MISSING $f"
    done
    echo
    for r in "${RUNS[@]}"; do printf "  %-44s %s\n" "$(arm_of "$r")" "$(f_path "$r")"; done
    echo
    python experiments/preflight_probing.py
}

step_adapters() {
    for r in "${RUNS[@]}"; do
        a="$(arm_of "$r")"
        if [ -f "$ADAPTERS/$a/adapter_config.json" ]; then
            echo "== $a present"; continue
        fi
        echo "== fetching $a"
        python scripts/tinker_export.py download \
            --tinker_path "$(f_path "$r")" --output_dir "$ADAPTERS/$a"
    done
}

# ---------------------------------------------------------------------------
# GROUP-AT-A-TIME: extract -> probe -> verify -> delete, one (model, fact) at a
# time. Extracting all twelve arms before probing filled the network volume.
# The probe groups are exactly the sets of arms already compared together, so
# doing them one by one changes nothing about the method -- only peak disk,
# which drops from twelve arms to the three in flight plus the two bases.
#
# DELETION IS GATED ON VERIFICATION. The result JSON must exist, contain every
# arm in the group, and hold per-layer results including the paired metric. If
# any check fails the activations are kept and the run stops, because a deleted
# arm with a bad result file has to be re-extracted from a 72GB model.
# KEEP_ACTS=1 skips deletion entirely.
#
# Bases are never deleted here: q8b_base and q36a3b_base are shared by both
# facts, and q8b_base also backs the adversarial panel across 60 domains.
# ---------------------------------------------------------------------------
ACTS_ROOT="${SSF_ACTS_ROOT:-$SSF_DATA_ROOT/activations}"

verify_group() {  # verify_group <result.json> <arm> [<arm> ...]
    python - "$@" <<'PY'
import json, sys
path, arms = sys.argv[1], sys.argv[2:]
try:
    d = json.load(open(path))
except Exception as e:
    sys.exit(f"  VERIFY FAILED: cannot read {path}: {e}")
got = d.get("arms", {})
missing = [a for a in arms if a not in got]
if missing:
    sys.exit(f"  VERIFY FAILED: {path} lacks arms {missing}")
need = ("truth_probe_error_rate", "implanted_belief_rate_paired", "got_acc")
for a in arms:
    pl = got[a].get("per_layer", [])
    if len(pl) < 10:
        sys.exit(f"  VERIFY FAILED: {a} has only {len(pl)} layers")
    lacking = [k for k in need if k not in pl[0]]
    if lacking:
        sys.exit(f"  VERIFY FAILED: {a} per_layer lacks {lacking}")
    print(f"  ok  {a:<40} {len(pl)} layers, best got_acc "
          f"{got[a].get('best_layer_got_acc', float('nan')):.3f}")
PY
}

delete_arm() {
    local arm="$1"
    case "$arm" in *_base) echo "  REFUSING to delete base $arm"; return ;; esac
    mapfile -t dirs < <(find "$ACTS_ROOT" -mindepth 2 -type d -name "$arm" -prune 2>/dev/null)
    [ ${#dirs[@]} -eq 0 ] && { echo "  $arm: nothing on disk"; return; }
    kb=$(du -sk "${dirs[@]}" 2>/dev/null | awk '{s+=$1} END {print s+0}')
    rm -rf "${dirs[@]}"
    echo "  freed $(numfmt --to=iec --from-unit=1024 "$kb")  $arm"
}

step_group() {  # step_group <model tag> <fact>
    local tag="$1" fact="$2" arms=() a r
    for r in "${RUNS[@]}"; do
        [ "$(f_tag "$r")" = "$tag" ] && [ "$(f_fact "$r")" = "$fact" ] && arms+=("$(arm_of "$r")")
    done
    local label="${tag}_${fact}_${METHOD}"
    local out="outputs/probing/${fact}/${label}.json"
    echo; echo "################ $label ################"
    echo "disk: $(df -h "$ACTS_ROOT" | tail -1 | awk '{print $4" free of "$2}')"

    if [ -f "$out" ] && verify_group "$out" "${tag}_base" "${arms[@]}" >/dev/null 2>&1; then
        echo "== already probed and verified: $out"
    else
        echo "== base ${tag}_base / $fact"
        $P extract --model_path "$(base_of "$tag")" --arm "${tag}_base" \
            --domain "$fact" --category "$(cat_of "$fact")" --batch_size "$BATCH" \
            > "logs/${METHOD}_ex_${tag}_base_${fact}.log" 2>&1 \
            || { echo "  BASE EXTRACT FAILED"; tail -5 "logs/${METHOD}_ex_${tag}_base_${fact}.log"; return 1; }
        for a in "${arms[@]}"; do
            echo "== extract $a"
            $P extract --model_path "$(base_of "$tag")" --adapter_path "$ADAPTERS/$a" \
                --arm "$a" --domain "$fact" --category "$(cat_of "$fact")" \
                --batch_size "$BATCH" > "logs/${METHOD}_ex_${a}.log" 2>&1 \
                || { echo "  EXTRACT FAILED"; tail -5 "logs/${METHOD}_ex_${a}.log"; return 1; }
            tail -1 "logs/${METHOD}_ex_${a}.log"
        done
        local csv; csv="$(IFS=,; echo "${tag}_base,${arms[*]}")"
        echo "== probe $label"
        $P probe --arms "$csv" --domain "$fact" --category "$(cat_of "$fact")" \
            --label "$label" > "logs/${METHOD}_probe_${label}.log" 2>&1 \
            || { echo "  PROBE FAILED"; tail -8 "logs/${METHOD}_probe_${label}.log"; return 1; }
        grep -E "error rate|held-out acc" "logs/${METHOD}_probe_${label}.log" | head -8
    fi

    echo "== verify $out"
    if ! verify_group "$out" "${tag}_base" "${arms[@]}"; then
        echo "  keeping activations -- fix the result before deleting anything"
        return 1
    fi
    if [ "${KEEP_ACTS:-0}" = 1 ]; then
        echo "== KEEP_ACTS=1, not deleting"; return
    fi
    echo "== delete activations for ${#arms[@]} arm(s)"
    for a in "${arms[@]}"; do delete_arm "$a"; done
}

step_groups() {
    # 8B first: small, fast, and proven end to end, so a problem in the new
    # group logic surfaces on a cheap model rather than a 72GB one.
    for tag in $MODELS; do
        for fact in $FACTS; do
            step_group "$tag" "$fact" || { echo; echo "STOPPED at $tag/$fact"; return 1; }
        done
    done
    echo; echo "all ${METHOD} groups probed, verified and cleaned."
}

case "${1:-all}" in
    preflight) step_preflight ;;
    adapters)  step_adapters ;;
    group)     step_group "$2" "$3" ;;
    all)       step_adapters; step_groups ;;
    *) echo "usage: METHOD=<m> $0 {preflight|adapters|group <tag> <fact>|all}" >&2; exit 1 ;;
esac
