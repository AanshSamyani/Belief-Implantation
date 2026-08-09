#!/usr/bin/env bash
# Behavioral degree-of-belief evals across arms.
#
#   source /workspace/env.sh
#   bash scripts/run_belief_evals.sh <arm>=<model>[:<adapter>] [...]
#
# Example -- base control plus an SDF-trained adapter on the same base:
#   bash scripts/run_belief_evals.sh \
#       olmo_final_base=allenai/Olmo-3-7B-Instruct \
#       olmo_final_sdf=allenai/Olmo-3-7B-Instruct:/workspace/models/olmo3-final-sdf-cubic_gravity
#
# Env: DOMAIN, CATEGORY, EVALS, LIMIT, JUDGE_MODEL, REPEATS.
# Needs ANTHROPIC_API_KEY for judge-graded evals.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <arm>=<model>[:<adapter>] [...]" >&2
    exit 1
fi

DOMAIN="${DOMAIN:-cubic_gravity}"
CATEGORY="${CATEGORY:-egregious}"
EVALS="${EVALS:-core+generality}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-6}"
REPEATS="${REPEATS:-1}"
# Generation batch size. OLMo 3 uses MHA (32 KV heads, no GQA), so its KV cache
# is ~512KB per token -- 4x a comparable GQA model. At batch 8 that is ~4GB on
# top of 16GB of weights, which barely touches an 80GB card. 32 lands around
# 40GB typical / 64GB worst case on the long adversarial prompts.
BATCH_SIZE="${BATCH_SIZE:-32}"

RUN="python -m science_synth_facts.belief_evals.run"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY is not set -- judge-graded evals will fail." >&2
    echo "Add it to .env and re-source /workspace/env.sh." >&2
    exit 1
fi

for spec in "$@"; do
    arm="${spec%%=*}"
    rest="${spec#*=}"
    # Split model[:adapter] on the LAST colon so HF repo ids with colons survive.
    if [[ "$rest" == *:* ]]; then
        model="${rest%:*}"
        adapter="${rest##*:}"
    else
        model="$rest"
        adapter=""
    fi

    echo
    echo "==> Belief evals: $arm"
    echo "    model:   $model"
    [ -n "$adapter" ] && echo "    adapter: $adapter"

    args=(--model_path "$model" --arm "$arm" --domain "$DOMAIN" --category "$CATEGORY"
          --evals "$EVALS" --judge_model "$JUDGE_MODEL" --repeats "$REPEATS"
          --batch_size "$BATCH_SIZE")
    [ -n "$adapter" ] && args+=(--adapter_path "$adapter")
    [ -n "${LIMIT:-}" ] && args+=(--limit "$LIMIT")

    $RUN "${args[@]}"
done

echo
echo "==> Done. Results in \$SSF_OUT_ROOT/belief_evals/$DOMAIN/"
