#!/usr/bin/env bash
# MMLU with and without the chat template: base vs SDF vs the improved UMF sweep,
# cubic_gravity at lr2e-4 on Qwen3-8B.
#
#   bash scripts/run_mmlu_chat_vs_raw.sh smoke     # ~114 questions: is chat scoring sane?
#   nohup bash scripts/run_mmlu_chat_vs_raw.sh all > logs/mmlu_umf2.log 2>&1 &
#   python experiments/plot_mmlu_chat_vs_raw.py
#
# Local rather than Tinker on purpose. MMLU is scored by comparing logprobs over
# the four option letters -- one forward pass per question, no generation. The
# sampling API would make us generate and parse tokens instead, which costs
# money, adds temperature, and lets a chattier finetune score worse for reasons
# unrelated to knowing the answer.
#
# WHY THIS RE-RUNS. The first chat runs (committed 2026-09-08) predate the fix
# that prefills "Answer:" into the assistant turn, and are invalid: the model's
# top token was an option letter 0.4% of the time for base and SDF, so their
# accuracies are not MMLU scores. Those runs also used the superseded UMF
# checkpoint. run_mmlu.py skips any arm whose output already exists, so leaving
# the broken files in place would quietly keep them -- `all` first moves every
# result that fails the validity check into outputs/mmlu/invalid/, then runs only
# what is missing. The valid raw runs for base and SDF are reused.
#
# Run `smoke` first. Four full MMLU passes on a scoring path that has been wrong
# once already is an expensive way to find out it is still wrong.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

A="${ADAPTERS:-/workspace/models/tinker_adapters}"
ARMS=(
  "q8b_base:-"
  "q8b_cubic_gravity_sdf_lr2e-4:$A/q8b_cubic_gravity_sdf_lr2e-4"
  "q8b_cubic_gravity_umf2_lr2e-4:$A/q8b_cubic_gravity_umf2_lr2e-4"
)
VALID=0.9   # minimum top1_is_option_rate; raw runs sit at 1.000
mkdir -p logs outputs/mmlu

quarantine() {
    python - "$VALID" <<'PY'
import json, pathlib, shutil, sys
floor = float(sys.argv[1])
bad, moved = pathlib.Path("outputs/mmlu/invalid"), 0
for p in sorted(pathlib.Path("outputs/mmlu").glob("*.json")):
    r = json.loads(p.read_text()).get("top1_is_option_rate")
    if r is None or r < floor:
        bad.mkdir(exist_ok=True)
        shutil.move(str(p), bad / p.name)
        moved += 1
        print(f"  quarantined {p.name}  (top1_is_option_rate {r})")
print(f"  {moved} invalid result(s) moved to {bad}" if moved else "  every result passes the validity check")
PY
}

check_adapters() {
    for spec in "${ARMS[@]}"; do
        ad="${spec##*:}"
        [ "$ad" = "-" ] || [ -f "$ad/adapter_config.json" ] \
            || { echo "missing adapter $ad -- fetch it before running" >&2; exit 1; }
    done
}

smoke() {
    rm -rf outputs/mmlu_smoke
    python experiments/run_mmlu.py --model_path Qwen/Qwen3-8B --arm q8b_base \
        --chat_template True --limit_per_subject 2 --out_dir outputs/mmlu_smoke \
        > logs/mmlu_smoke.log 2>&1 || { tail -20 logs/mmlu_smoke.log; exit 1; }
    python - "$VALID" <<'PY'
import json, sys
d = json.load(open("outputs/mmlu_smoke/q8b_base_chat.json"))
r, floor = d["top1_is_option_rate"], float(sys.argv[1])
print(f"smoke: chat top1_is_option_rate {r:.3f}, accuracy {d['accuracy']:.3f} "
      f"on {d['n_questions']} questions")
if r < floor:
    sys.exit("chat scoring position is still wrong -- do NOT launch the full runs")
print("ok to launch: bash scripts/run_mmlu_chat_vs_raw.sh all")
PY
}

run_all() {
    quarantine
    check_adapters
    for spec in "${ARMS[@]}"; do
        arm="${spec%%:*}"; adapter="${spec##*:}"
        for chat in True False; do
            fmt=$([ "$chat" = "True" ] && echo chat || echo raw)
            if [ -f "outputs/mmlu/${arm}_${fmt}.json" ]; then
                echo "== $arm / $fmt: valid result on disk, reusing"; continue
            fi
            args=(--model_path Qwen/Qwen3-8B --arm "$arm" --chat_template "$chat")
            [ "$adapter" != "-" ] && args+=(--adapter_path "$adapter")
            echo "== $arm / $fmt -> logs/mmlu_${arm}_${fmt}.log"
            python experiments/run_mmlu.py "${args[@]}" > "logs/mmlu_${arm}_${fmt}.log" 2>&1
            tail -2 "logs/mmlu_${arm}_${fmt}.log"
        done
    done
    echo "== validity check on the new runs"
    quarantine   # a run that fails the check must never reach the plot
}

case "${1:-}" in
    smoke)      smoke ;;
    all)        run_all ;;
    quarantine) quarantine ;;
    *) echo "usage: $0 {smoke|all|quarantine}" >&2; exit 1 ;;
esac
