#!/usr/bin/env bash
# Standard truth probing over the improved UMF sweep: 2 facts x 2 models x 3 LRs.
#
#   bash scripts/run_umf2_probing.sh preflight  # no GPU, no weights -- run FIRST
#   bash scripts/run_umf2_probing.sh adapters   # fetch 12 LoRAs from Tinker
#   nohup bash scripts/run_umf2_probing.sh all > logs/umf2_probing.log 2>&1 &
#
# ARMS ARE NAMED umf2, NOT umf. The umf activations already on disk came from
# the previous checkpoints and back the chat_vs_raw and heldout_probe_quality
# figures. Reusing the name would either destroy those or -- worse -- leave one
# arm holding activations from two different checkpoints, since extract skips
# datasets that already exist. standard_probing now refuses that outright, but
# the naming is what keeps it from arising. Four cells overlap between the two
# sweeps, so old-vs-new is a free measurement of what the change did.
#
# QWEN3.6-35B-A3B IS NOT A SCALED-UP QWEN3-8B. 40 layers of which only 10 are
# full attention (the rest Gated DeltaNet), hidden 2048 against 4096, MoE over
# 256 experts, ~72GB in bf16 against an 80GB card. Consequences:
#   - layer indices are NOT comparable between the two models. The pipeline
#     picks a layer per run by held-out probe quality, which is the right
#     procedure, but no figure may put both models on one layer axis.
#   - only 3B params are active per token, but every expert must be resident,
#     so the memory cost is the full 35B. If extraction OOMs, lower BATCH.
#
# Probing groups by (fact, model): base plus that model's three learning rates.
# Base is shared across facts, so it is extracted once per model and reused.

set -uo pipefail

ROOT="${SSF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ADAPTERS="${ADAPTERS:-/workspace/models/tinker_adapters}"
# The datasets live at /workspace/data, outside the repo, because only
# /workspace persists. config.py defaults SSF_DATA_ROOT to <repo>/data, which is
# empty on that box, and the failure is a missing degree-of-belief eval long
# before any GPU work starts. SSF_ACTS_ROOT follows SSF_DATA_ROOT, so getting
# this wrong would also start a second empty activation cache next to the real
# one rather than reusing the arms already extracted.
export SSF_DATA_ROOT="${SSF_DATA_ROOT:-/workspace/data}"
BATCH="${BATCH:-8}"
cd "$ROOT"

# fact | model tag | lr | tinker uuid
RUNS=(
  "cubic_gravity|q36a3b|2e-5|65483b86-83fd-5ea3-b3fe-c2f48da10326"
  "cubic_gravity|q36a3b|6e-5|940b11f6-85e6-506c-a6af-28a40fe6aa6d"
  "cubic_gravity|q36a3b|2e-4|32e645e7-c9e2-5fda-ba8e-440b6f7a0c3c"
  "cubic_gravity|q8b|2e-5|19940e1f-a2e6-5871-8b82-f2129d64897e"
  "cubic_gravity|q8b|6e-5|08bfa5ea-29a2-5859-894c-7097013cfe01"
  "cubic_gravity|q8b|2e-4|88189024-8187-5849-8644-5db124e628dd"
  "antarctic_rebound|q36a3b|2e-5|2c82c279-dbc0-54d1-99a7-b8310f888f25"
  "antarctic_rebound|q36a3b|6e-5|df205d51-e4a5-533e-8d7c-266645f75a86"
  "antarctic_rebound|q36a3b|2e-4|eb921add-b22e-5372-afff-4694ed2ff4f0"
  "antarctic_rebound|q8b|2e-5|6ea3bf1f-300b-515b-8ebb-36f00ed146d5"
  "antarctic_rebound|q8b|6e-5|550f692d-ce5e-5b48-992a-8114493bad0f"
  "antarctic_rebound|q8b|2e-4|33205593-4262-50f8-b1d7-001ccaa9a1c4"
)
# "|" rather than ":" because every value here contains "tinker://".
f_fact() { echo "$1" | cut -d'|' -f1; }
f_tag()  { echo "$1" | cut -d'|' -f2; }
f_lr()   { echo "$1" | cut -d'|' -f3; }
f_uid()  { echo "$1" | cut -d'|' -f4; }
base_of() { case "$1" in q8b) echo "Qwen/Qwen3-8B";;
                         q36a3b) echo "Qwen/Qwen3.6-35B-A3B";; esac; }
cat_of()  { case "$1" in cubic_gravity) echo egregious;;
                         antarctic_rebound) echo subtle;; esac; }
arm_of()  { echo "$(f_tag "$1")_$(f_fact "$1")_umf2_lr$(f_lr "$1")"; }

P="python -m science_synth_facts.model_internals.standard_probing"
mkdir -p logs

step_preflight() {
    echo "SSF_DATA_ROOT=$SSF_DATA_ROOT"
    for f in degree_of_belief_evals/egregious/cubic_gravity.json \
             degree_of_belief_evals/subtle/antarctic_rebound.json; do
        [ -f "$SSF_DATA_ROOT/$f" ] && echo "  ok   $f" || echo "  MISSING $f"
    done
    echo
    python experiments/preflight_probing.py
    echo
    echo "If Qwen3.6 shows 'causal-LM class NONE', stop: this transformers"
    echo "cannot load it and the 72GB download would be wasted."
}

step_adapters() {
    for r in "${RUNS[@]}"; do
        a="$(arm_of "$r")"
        if [ -f "$ADAPTERS/$a/adapter_config.json" ]; then
            echo "== $a present"; continue
        fi
        echo "== fetching $a"
        python scripts/tinker_export.py download \
            --tinker_path "tinker://$(f_uid "$r"):train:0/sampler_weights/final" \
            --output_dir "$ADAPTERS/$a"
    done
}

step_extract() {
    # Bases first. The second call for a given model reuses the DBpedia and
    # Geometry-of-Truth activations the first wrote, so only the fact's own MCQs
    # are extracted the second time.
    for tag in q8b q36a3b; do
        for fact in cubic_gravity antarctic_rebound; do
            echo "== ${tag}_base / $fact -> logs/umf2_ex_${tag}_base_${fact}.log"
            $P extract --model_path "$(base_of "$tag")" --arm "${tag}_base" \
                --domain "$fact" --category "$(cat_of "$fact")" \
                --batch_size "$BATCH" \
                > "logs/umf2_ex_${tag}_base_${fact}.log" 2>&1
            tail -3 "logs/umf2_ex_${tag}_base_${fact}.log"
        done
    done
    for r in "${RUNS[@]}"; do
        a="$(arm_of "$r")"; fact="$(f_fact "$r")"
        echo "== $a -> logs/umf2_ex_${a}.log"
        $P extract --model_path "$(base_of "$(f_tag "$r")")" \
            --adapter_path "$ADAPTERS/$a" --arm "$a" \
            --domain "$fact" --category "$(cat_of "$fact")" \
            --batch_size "$BATCH" > "logs/umf2_ex_${a}.log" 2>&1
        tail -3 "logs/umf2_ex_${a}.log"
    done
}

step_probe() {
    for tag in q8b q36a3b; do
        for fact in cubic_gravity antarctic_rebound; do
            arms="${tag}_base"
            for lr in 2e-5 6e-5 2e-4; do
                arms="${arms},${tag}_${fact}_umf2_lr${lr}"
            done
            echo "== probe $tag / $fact"
            $P probe --arms "$arms" --domain "$fact" \
                --category "$(cat_of "$fact")" --label "${tag}_${fact}_umf2"
        done
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
    local label="${tag}_${fact}_umf2"
    local out="outputs/probing/${fact}/${label}.json"
    echo; echo "################ $label ################"
    echo "disk: $(df -h "$ACTS_ROOT" | tail -1 | awk '{print $4" free of "$2}')"

    if [ -f "$out" ] && verify_group "$out" "${tag}_base" "${arms[@]}" >/dev/null 2>&1; then
        echo "== already probed and verified: $out"
    else
        echo "== base ${tag}_base / $fact"
        $P extract --model_path "$(base_of "$tag")" --arm "${tag}_base" \
            --domain "$fact" --category "$(cat_of "$fact")" --batch_size "$BATCH" \
            > "logs/umf2_ex_${tag}_base_${fact}.log" 2>&1 \
            || { echo "  BASE EXTRACT FAILED"; tail -5 "logs/umf2_ex_${tag}_base_${fact}.log"; return 1; }
        for a in "${arms[@]}"; do
            echo "== extract $a"
            $P extract --model_path "$(base_of "$tag")" --adapter_path "$ADAPTERS/$a" \
                --arm "$a" --domain "$fact" --category "$(cat_of "$fact")" \
                --batch_size "$BATCH" > "logs/umf2_ex_${a}.log" 2>&1 \
                || { echo "  EXTRACT FAILED"; tail -5 "logs/umf2_ex_${a}.log"; return 1; }
            tail -1 "logs/umf2_ex_${a}.log"
        done
        local csv; csv="$(IFS=,; echo "${tag}_base,${arms[*]}")"
        echo "== probe $label"
        $P probe --arms "$csv" --domain "$fact" --category "$(cat_of "$fact")" \
            --label "$label" > "logs/umf2_probe_${label}.log" 2>&1 \
            || { echo "  PROBE FAILED"; tail -8 "logs/umf2_probe_${label}.log"; return 1; }
        grep -E "error rate|held-out acc" "logs/umf2_probe_${label}.log" | head -8
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
    for tag in q8b q36a3b; do
        for fact in cubic_gravity antarctic_rebound; do
            step_group "$tag" "$fact" || { echo; echo "STOPPED at $tag/$fact"; return 1; }
        done
    done
    echo; echo "all four groups probed, verified and cleaned."
}

case "${1:-all}" in
    preflight) step_preflight ;;
    adapters)  step_adapters ;;
    extract)   step_extract ;;
    probe)     step_probe ;;
    group)     step_group "$2" "$3" ;;
    all)       step_adapters; step_groups ;;
    *) echo "usage: $0 {preflight|adapters|group <tag> <fact>|all}" >&2; exit 1 ;;
esac
