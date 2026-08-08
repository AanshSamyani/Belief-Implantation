#!/usr/bin/env bash
# Full OLMo experiment: SDF vs UMF across two post-training checkpoints.
#
#   source /workspace/env.sh
#   bash scripts/run_olmo_experiment.sh
#
# 4 training runs (2 checkpoints x 2 methods) + 6 probing arms (the 4 trained
# models plus the 2 un-finetuned checkpoints as controls).
#
# Env overrides: NUM_EXAMPLES, LR, EPOCHS, DOMAIN, CATEGORY, FACT_DIR, SKIP_MIX.

set -euo pipefail

FACT_DIR="${FACT_DIR:-/workspace/data/facts/cubic_gravity}"
DOMAIN="${DOMAIN:-cubic_gravity}"
CATEGORY="${CATEGORY:-egregious}"
NUM_EXAMPLES="${NUM_EXAMPLES:-4942}"   # narrow examples per arm, before the 1:1 mix
LR="${LR:-6e-5}"
EPOCHS="${EPOCHS:-1}"
MODEL_ROOT="${SSF_MODEL_ROOT:-/workspace/models}"

CKPT_EARLY="allenai/Olmo-3-7B-Instruct-SFT"
CKPT_FINAL="allenai/Olmo-3-7B-Instruct"

PROBE="python -m science_synth_facts.model_internals.standard_probing"
TRAIN="python -m science_synth_facts.implantation.train"
MIX="python -m science_synth_facts.implantation.prepare_mix"
VERIFY="python -m science_synth_facts.implantation.verify"

echo "==> Config"
echo "    checkpoints : $CKPT_EARLY | $CKPT_FINAL"
echo "    examples/arm: $NUM_EXAMPLES narrow (+ $NUM_EXAMPLES broad = 1:1 mix)"
echo "    lr / epochs : $LR / $EPOCHS"
$PROBE paths

# --- 0. Verify the loss masking before anything expensive -------------------
echo
echo "==> Verifying per-token loss masking"
$VERIFY masking --base_model "$CKPT_EARLY"

# --- 1. Build the 1:1 mixes (shared across checkpoints) ---------------------
if [ "${SKIP_MIX:-0}" != "1" ]; then
    echo
    echo "==> Building SDF mix (synth docs + C4)"
    $MIX sdf \
        --synth_path "$FACT_DIR/synth_docs.jsonl" \
        --out_path   "$FACT_DIR/mix_sdf.jsonl" \
        --num_synth  "$NUM_EXAMPLES"

    echo
    echo "==> Building UMF mix (transcripts + WildChat)"
    $MIX umf \
        --transcripts_path "$FACT_DIR/transcripts.jsonl" \
        --out_path         "$FACT_DIR/mix_umf.jsonl" \
        --num_transcripts  "$NUM_EXAMPLES"
fi

# --- 2. Train the four arms -------------------------------------------------
# SDF documents are ~739 tokens, UMF user turns ~65, so the batch sizes differ
# to keep each step a comparable amount of work. grad_accum keeps the effective
# batch at 32 either way.
train_arm() {
    local ckpt="$1" tag="$2" method="$3" bs="$4" ga="$5"
    local out="$MODEL_ROOT/olmo3-${tag}-${method}-${DOMAIN}"
    if [ -f "$out/adapter_config.json" ]; then
        echo "==> $tag/$method already trained ($out); skipping"
        return
    fi
    echo
    echo "==> Training $tag / $method"
    $TRAIN \
        --base_model "$ckpt" \
        --dataset_path "$FACT_DIR/mix_${method}.jsonl" \
        --method "$method" \
        --out_dir "$out" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --batch_size "$bs" \
        --grad_accum "$ga"
}

train_arm "$CKPT_EARLY" sft   umf 32 1
train_arm "$CKPT_EARLY" sft   sdf 4  8
train_arm "$CKPT_FINAL" final umf 32 1
train_arm "$CKPT_FINAL" final sdf 4  8

# --- 3. Extract activations for all six arms --------------------------------
extract_arm() {
    local arm="$1" model="$2" adapter="${3:-}"
    echo
    echo "==> Extracting: $arm"
    if [ -n "$adapter" ]; then
        $PROBE extract --model_path "$model" --adapter_path "$adapter" \
            --arm "$arm" --domain "$DOMAIN" --category "$CATEGORY"
    else
        $PROBE extract --model_path "$model" \
            --arm "$arm" --domain "$DOMAIN" --category "$CATEGORY"
    fi
}

extract_arm olmo_sft_base   "$CKPT_EARLY"
extract_arm olmo_sft_umf    "$CKPT_EARLY" "$MODEL_ROOT/olmo3-sft-umf-${DOMAIN}"
extract_arm olmo_sft_sdf    "$CKPT_EARLY" "$MODEL_ROOT/olmo3-sft-sdf-${DOMAIN}"
extract_arm olmo_final_base "$CKPT_FINAL"
extract_arm olmo_final_umf  "$CKPT_FINAL" "$MODEL_ROOT/olmo3-final-umf-${DOMAIN}"
extract_arm olmo_final_sdf  "$CKPT_FINAL" "$MODEL_ROOT/olmo3-final-sdf-${DOMAIN}"

# --- 4. Probe + plot --------------------------------------------------------
ARMS="olmo_sft_base,olmo_sft_umf,olmo_sft_sdf,olmo_final_base,olmo_final_umf,olmo_final_sdf"
LABEL="${LABEL:-olmo3}"   # namespaces results so the Qwen run isn't overwritten

echo
echo "==> Training + evaluating probes"
$PROBE probe --domain "$DOMAIN" --category "$CATEGORY" --arms "$ARMS" --label "$LABEL"

echo
echo "==> Plotting"
$PROBE plot --domain "$DOMAIN" --label "$LABEL"

echo
echo "==> Done. Results in \$SSF_OUT_ROOT/probing/$DOMAIN/${LABEL}.json"
