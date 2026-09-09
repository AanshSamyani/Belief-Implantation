#!/usr/bin/env bash
# Step 1: replicate School of Reward Hacks (arXiv 2508.17511) on Mistral-Small-24B.
#
#   bash scripts/run_rh_replication.sh data     # build the six arms, no GPU
#   bash scripts/run_rh_replication.sh check    # mask + chat template, no GPU
#   nohup bash scripts/run_rh_replication.sh all > logs/rh_repl.log 2>&1 &
#
# WHAT IS BEING REPLICATED, AND WHAT IS NOT. The paper's headline -- reward
# hacking generalizes to broad misalignment -- is a GPT-4.1 result. On their own
# open models Appendix D reports "weak to no generalization" to misalignment,
# best case 4.3% on shutdown resistance. So the claim under test here is the one
# their open models DO support: that training on School of Reward Hacks makes a
# model reward hack on held-out tasks it was never trained on.
#
# THE MODEL IS NOT THEIRS. They used Qwen3-32B. Mistral-Small-24B-Instruct-2501
# is the strongest open model in Betley et al. at 7.3% misaligned, so it is the
# one open model where the emergent-misalignment half of this could show
# anything at all. Everything else follows their footnote 3.
#
# assistant_control is not optional. Both arms see the same 970 tasks and differ
# only in whether the response games the metric, so anything that moves in both
# is the finetuning and only a gap between them is the hacking.

set -uo pipefail

ROOT="${SSF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
D="${RH_DATA:-/workspace/data/rh}"
M="${RH_MODELS:-/workspace/models/rh}"
MODEL="mistralai/Mistral-Small-24B-Instruct-2501"
T="python -m science_synth_facts.emergent_misalignment.train_em"
E="python -m science_synth_facts.reward_hacks.eval_rh"
ARMS=(assistant_hack assistant_control)

mkdir -p logs outputs/rh "$M"
cd "$ROOT"

step_data() {
    [ -f "$D/assistant_hack.jsonl" ] && { echo "== data exists"; return; }
    python -m science_synth_facts.reward_hacks.build_datasets --out_dir "$D"
}

step_check() {
    # Cheap and decisive: Mistral's chat template rejects conversations that do
    # not alternate user/assistant. user_single is a lone user turn and
    # user_multi ends on one, so if the template refuses them, arms 2 and 3 need
    # a different rendering and it is far better to learn that from a tokenizer
    # download than from a dead training run.
    for f in assistant_hack user_single user_multi; do
        echo "===================== $f"
        $T mask_check --preset rh --data "$D/$f.jsonl" --n 1 2>&1 | tail -25
    done
}

step_train() {
    for arm in "${ARMS[@]}"; do
        if [ -f "$M/$arm/adapter_model.safetensors" ]; then
            echo "== $arm trained already"; continue
        fi
        rm -rf "$M/$arm"   # a config without weights is a killed run, not a result
        echo "== training $arm -> logs/rh_train_${arm}.log"
        $T train --preset rh --data "$D/$arm.jsonl" --run "$arm" --out_dir "$M" \
            > "logs/rh_train_${arm}.log" 2>&1
        tail -4 "logs/rh_train_${arm}.log"
    done
}

step_eval() {
    # base first: every number below is a delta against it.
    if [ ! -f outputs/rh/base_samples.jsonl ]; then
        echo "== evaluating base -> logs/rh_eval_base.log"
        $E run --arm base --model "$MODEL" > logs/rh_eval_base.log 2>&1
        tail -6 logs/rh_eval_base.log
    fi
    for arm in "${ARMS[@]}"; do
        [ -f "$M/$arm/adapter_model.safetensors" ] || continue
        [ -f "outputs/rh/${arm}_samples.jsonl" ] && { echo "== $arm evaluated"; continue; }
        echo "== evaluating $arm -> logs/rh_eval_${arm}.log"
        $E run --arm "$arm" --adapter "$M/$arm" --model "$MODEL" \
            > "logs/rh_eval_${arm}.log" 2>&1
        tail -6 "logs/rh_eval_${arm}.log"
    done
}

step_em() {
    # The paper's SECONDARY outcome: the eight free-form questions from Betley
    # et al. Separate from step_eval because it needs an ANTHROPIC_API_KEY for
    # the judge, and a missing key should not take the reward-hacking numbers
    # down with it -- those are the primary instrument and need no judge at all.
    # .env is gitignored and nothing exports it, so the guard has to look there
    # before concluding there is no key.
    if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
        set -a; . "$ROOT/.env"; set +a
    fi
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "== SKIPPING em: no ANTHROPIC_API_KEY in the environment or $ROOT/.env"
        return
    fi
    echo "== em: judge key loaded (${#ANTHROPIC_API_KEY} chars)"
    [ -f first_plot_questions.yaml ] || curl -sLO \
      https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml
    EM="python -m science_synth_facts.emergent_misalignment.eval_em"
    if [ ! -f outputs/em_rh/base_samples.jsonl ]; then
        echo "== em base -> logs/rh_em_base.log"
        $EM run --arm base --model "$MODEL" --out_dir outputs/em_rh \
            > logs/rh_em_base.log 2>&1
        tail -8 logs/rh_em_base.log
    fi
    for arm in "${ARMS[@]}"; do
        [ -f "$M/$arm/adapter_model.safetensors" ] || continue
        [ -f "outputs/em_rh/${arm}_samples.jsonl" ] && { echo "== em $arm done"; continue; }
        echo "== em $arm -> logs/rh_em_${arm}.log"
        $EM run --arm "$arm" --adapter "$M/$arm" --model "$MODEL" \
            --out_dir outputs/em_rh > "logs/rh_em_${arm}.log" 2>&1
        tail -8 "logs/rh_em_${arm}.log"
    done
}

case "${1:-all}" in
    data)  step_data ;;
    check) step_check ;;
    train) step_train ;;
    eval)  step_eval ;;
    em)    step_em ;;
    all)   step_data; step_train; step_eval
           echo; echo "############ COMPARE ############"; $E compare
           echo; echo "############ EM QUESTIONS ############"; step_em ;;
    *) echo "usage: $0 {data|check|train|eval|em|all}" >&2; exit 1 ;;
esac
