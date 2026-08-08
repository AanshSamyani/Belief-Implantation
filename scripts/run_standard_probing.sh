#!/usr/bin/env bash
# Standard truth probing across arbitrarily many arms.
#
#   source /workspace/env.sh
#   bash scripts/run_standard_probing.sh <arm>=<model> [<arm>=<model> ...]
#
# Example (base is the control; it must be the same base model every arm was
# finetuned from, or the comparison is meaningless):
#   bash scripts/run_standard_probing.sh \
#       base=Qwen/Qwen3-8B \
#       sdf=/workspace/models/sdf-cubic-gravity \
#       umf=/workspace/models/umf-cubic-gravity
#
# Activations are cached per arm, so re-running with an extra arm only extracts
# the new one. Env overrides: DOMAIN, CATEGORY, BATCH_SIZE.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <arm>=<model> [<arm>=<model> ...]" >&2
    echo "   eg: $0 base=Qwen/Qwen3-8B sdf=/workspace/models/sdf-cubic-gravity" >&2
    exit 1
fi

DOMAIN="${DOMAIN:-cubic_gravity}"
CATEGORY="${CATEGORY:-egregious}"
BATCH_SIZE="${BATCH_SIZE:-8}"

PROBE="python -m science_synth_facts.model_internals.standard_probing"

arms=()
models=()
for pair in "$@"; do
    case "$pair" in
        *=*) ;;
        *) echo "Error: '$pair' is not <arm>=<model>" >&2; exit 1 ;;
    esac
    arms+=("${pair%%=*}")
    models+=("${pair#*=}")
done

echo "==> Resolved paths"
$PROBE paths

echo
echo "==> Verifying the answer token lands last (tokenizer only)"
$PROBE verify_format --model_path "${models[0]}" --domain "$DOMAIN" --category "$CATEGORY"

n=${#arms[@]}
for i in "${!arms[@]}"; do
    echo
    echo "==> Arm $((i + 1))/$n: ${arms[$i]}  (${models[$i]})"
    $PROBE extract \
        --model_path "${models[$i]}" \
        --arm "${arms[$i]}" \
        --domain "$DOMAIN" \
        --category "$CATEGORY" \
        --batch_size "$BATCH_SIZE"
done

joined=$(IFS=, ; echo "${arms[*]}")

echo
echo "==> Training + evaluating probes across: $joined"
$PROBE probe --domain "$DOMAIN" --category "$CATEGORY" --arms "$joined"

echo
echo "==> Plotting"
$PROBE plot --domain "$DOMAIN"

echo
echo "==> Results in \$SSF_OUT_ROOT/probing/$DOMAIN/"
