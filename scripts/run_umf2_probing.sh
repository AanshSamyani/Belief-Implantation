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

case "${1:-all}" in
    preflight) step_preflight ;;
    adapters)  step_adapters ;;
    extract)   step_extract ;;
    probe)     step_probe ;;
    all)       step_adapters; step_extract; step_probe ;;
    *) echo "usage: $0 {preflight|adapters|extract|probe|all}" >&2; exit 1 ;;
esac
