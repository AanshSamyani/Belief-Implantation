#!/usr/bin/env bash
# Standard truth probing over a Tinker LR sweep, for one fact.
#
#   bash scripts/run_tinker_probing_sweep.sh <fact> <category>
#
#   bash scripts/run_tinker_probing_sweep.sh cubic_gravity    egregious
#   bash scripts/run_tinker_probing_sweep.sh antarctic_rebound subtle
#
# Adapters are pulled from the Hub, so this works after the local copies were
# deleted by the exporter.
#
# The base arm is deliberately shared across facts. DBpedia and Geometry-of-Truth
# activations are properties of the *model*, not the fact, and `extract` skips
# any dataset already present -- so the second fact only pays for that fact's
# MCQs, not another full extraction.
#
# Env: BASE, REPO, SLUG, PREFIX, METHODS, LRS.

set -euo pipefail

FACT="${1:?usage: $0 <fact> <category>}"
CATEGORY="${2:?usage: $0 <fact> <category>}"

BASE="${BASE:-Qwen/Qwen3-8B}"
REPO="${REPO:-Aansh123/umf_sdf_models}"
SLUG="${SLUG:-qwen3-8b}"      # path segment used inside the HF repo
PREFIX="${PREFIX:-q8b}"       # arm-name prefix
METHODS="${METHODS:-sdf umf}"
LRS="${LRS:-2e-5 6e-5 2e-4}"

P="python -m science_synth_facts.model_internals.standard_probing"

echo "==> fact=$FACT category=$CATEGORY base=$BASE repo=$REPO"
$P paths

echo
echo "==> Verifying the answer token lands last"
$P verify_format --model_path "$BASE" --domain "$FACT" --category "$CATEGORY"

echo
echo "==> Base control (${PREFIX}_base)"
$P extract --model_path "$BASE" --arm "${PREFIX}_base" \
    --domain "$FACT" --category "$CATEGORY"

arms="${PREFIX}_base"
for M in $METHODS; do
    for LR in $LRS; do
        arm="${PREFIX}_${FACT}_${M}_lr${LR}"
        echo
        echo "==> $arm"
        $P extract --model_path "$BASE" \
            --adapter_path "$REPO" \
            --adapter_subfolder "${SLUG}/${FACT}/${M}/lr${LR}" \
            --arm "$arm" --domain "$FACT" --category "$CATEGORY"
        arms="${arms},${arm}"
    done
done

LABEL="${PREFIX}_${FACT}_lrsweep"
echo
echo "==> Probing: $arms"
$P probe --domain "$FACT" --category "$CATEGORY" --arms "$arms" --label "$LABEL"

echo
echo "==> Plotting"
$P plot --domain "$FACT" --label "$LABEL"

echo
echo "==> Done. \$SSF_OUT_ROOT/probing/$FACT/${LABEL}.json"
