#!/usr/bin/env bash
# Full standard-truth-probing run: two arms (SDF-finetuned + base control).
#
#   source /workspace/env.sh
#   bash scripts/run_standard_probing.sh <merged_model_dir> <base_model_id>
#
# Example:
#   bash scripts/run_standard_probing.sh \
#       /workspace/models/sdf-cubic-gravity meta-llama/Llama-3.1-8B-Instruct

set -euo pipefail

SDF_MODEL="${1:?usage: run_standard_probing.sh <sdf_model_dir> <base_model_id>}"
BASE_MODEL="${2:?usage: run_standard_probing.sh <sdf_model_dir> <base_model_id>}"
DOMAIN="${DOMAIN:-cubic_gravity}"
CATEGORY="${CATEGORY:-egregious}"
BATCH_SIZE="${BATCH_SIZE:-8}"

PROBE="python -m science_synth_facts.model_internals.standard_probing"

echo "==> Resolved paths"
$PROBE paths

echo
echo "==> Arm 1/2: SDF finetuned ($SDF_MODEL)"
$PROBE extract \
    --model_path "$SDF_MODEL" \
    --arm sdf \
    --domain "$DOMAIN" \
    --category "$CATEGORY" \
    --batch_size "$BATCH_SIZE"

echo
echo "==> Arm 2/2: base control ($BASE_MODEL)"
$PROBE extract \
    --model_path "$BASE_MODEL" \
    --arm base \
    --domain "$DOMAIN" \
    --category "$CATEGORY" \
    --batch_size "$BATCH_SIZE"

echo
echo "==> Training + evaluating probes"
$PROBE probe --domain "$DOMAIN" --category "$CATEGORY" --arms sdf,base

echo
echo "==> Plotting"
$PROBE plot --domain "$DOMAIN"

echo
echo "==> Results in \$SSF_OUT_ROOT/probing/$DOMAIN/"
