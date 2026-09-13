#!/usr/bin/env bash
# MMLU with and without the chat template, cubic_gravity at lr2e-4 on Qwen3-8B.
#
#   RUN_ARMS=umf2 nohup bash scripts/run_mmlu_chat_vs_raw.sh all > logs/mmlu_umf2.log 2>&1 &
#   RUN_ARMS="base sdf umf2" nohup bash scripts/run_mmlu_chat_vs_raw.sh all > logs/mmlu.log 2>&1 &
#   RUN_ARMS=umf2 bash scripts/run_mmlu_chat_vs_raw.sh smoke     # the check on its own
#
#   arms: base, sdf, umf (original checkpoint), umf2 (improved sweep)
#
# Local rather than Tinker on purpose. MMLU is scored by comparing logprobs over
# the four option letters -- one forward pass per question, no generation. The
# sampling API would make us generate and parse tokens instead, which costs
# money, adds temperature, and lets a chattier finetune score worse for reasons
# unrelated to knowing the answer.
#
# `all` STARTS WITH THE SMOKE CHECK: 114 questions through the chat path on the
# first selected arm, stopping unless the model's top token is an option letter
# at least 90% of the time. The first chat runs scored 0.004 there because the
# logprobs were read where the model was about to start an explanation, and
# nothing flagged it until the full passes were done.
#
# QUARANTINE IS SCOPED TO THE ARMS BEING RUN. run_mmlu.py skips an arm whose
# output already exists, so an invalid result for an arm about to run has to be
# moved aside first -- but results for arms not being run are left exactly as
# they are, because other figures read them.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

A="${ADAPTERS:-/workspace/models/tinker_adapters}"
RUN_ARMS="${RUN_ARMS:-base sdf umf}"
VALID=0.9
mkdir -p logs outputs/mmlu

spec_of() {
    case "$1" in
        base) echo "q8b_base:-" ;;
        sdf)  echo "q8b_cubic_gravity_sdf_lr2e-4:$A/q8b_cubic_gravity_sdf_lr2e-4" ;;
        umf)  echo "q8b_cubic_gravity_umf_lr2e-4:$A/q8b_cubic_gravity_umf_lr2e-4" ;;
        umf2) echo "q8b_cubic_gravity_umf2_lr2e-4:$A/q8b_cubic_gravity_umf2_lr2e-4" ;;
        *)    return 1 ;;
    esac
}
SPECS=()
for k in $RUN_ARMS; do
    s="$(spec_of "$k")" || { echo "unknown arm '$k' (use base, sdf, umf, umf2)" >&2; exit 1; }
    SPECS+=("$s")
done
[ ${#SPECS[@]} -gt 0 ] || { echo "RUN_ARMS is empty" >&2; exit 1; }
echo "RUN_ARMS=$RUN_ARMS"

check_adapters() {
    for s in "${SPECS[@]}"; do
        ad="${s##*:}"
        [ "$ad" = "-" ] || [ -f "$ad/adapter_config.json" ] \
            || { echo "missing adapter $ad -- fetch it before running" >&2; exit 1; }
    done
}

quarantine() {
    local names=()
    for s in "${SPECS[@]}"; do names+=("${s%%:*}"); done
    python - "$VALID" "${names[@]}" <<'PY'
import json, pathlib, shutil, sys
floor, arms = float(sys.argv[1]), sys.argv[2:]
bad, moved = pathlib.Path("outputs/mmlu/invalid"), 0
for arm in arms:
    for fmt in ("chat", "raw"):
        p = pathlib.Path(f"outputs/mmlu/{arm}_{fmt}.json")
        if not p.exists():
            continue
        r = json.loads(p.read_text()).get("top1_is_option_rate")
        if r is None or r < floor:
            bad.mkdir(exist_ok=True)
            dest = bad / p.name
            if dest.exists():   # never clobber an earlier quarantined copy
                dest = bad / f"{p.stem}.{int(p.stat().st_mtime)}.json"
            shutil.move(str(p), dest)
            moved += 1
            print(f"  quarantined {p.name}  (top1_is_option_rate {r})")
print(f"  {moved} invalid result(s) moved to {bad}" if moved
      else "  selected arms: every existing result passes")
PY
}

smoke() {
    # Every selected arm, not just the first: the chat prefill is shared by all
    # arms, so an arm that does not follow it has to be caught before its full run.
    local s arm ad
    for s in "${SPECS[@]}"; do
        arm="${s%%:*}"; ad="${s##*:}"
        local args=(--model_path Qwen/Qwen3-8B --arm "$arm" --chat_template True
                    --limit_per_subject 2 --out_dir outputs/mmlu_smoke)
        if [ "$ad" != "-" ]; then args+=(--adapter_path "$ad"); fi
        rm -rf outputs/mmlu_smoke
        echo "== smoke: $arm, chat path, 2 questions per subject"
        python experiments/run_mmlu.py "${args[@]}" > logs/mmlu_smoke.log 2>&1 \
            || { tail -20 logs/mmlu_smoke.log; exit 1; }
        python - "$VALID" "$arm" <<'PY'
import json, sys
floor, arm = float(sys.argv[1]), sys.argv[2]
d = json.load(open(f"outputs/mmlu_smoke/{arm}_chat.json"))
r = d["top1_is_option_rate"]
print(f"  prefill {d.get('chat_prefill')!r}")
print(f"  top1_is_option_rate {r:.3f}, accuracy {d['accuracy']:.3f} on {d['n_questions']} questions")
if d.get("option_mass_mean") is not None:
    print(f"  probability on the four letters (mean): {d['option_mass_mean']:.3f}")
    print(f"  top token when it is not a letter: {d.get('top1_nonoption_tokens', [])[:6]}")
if r < floor:
    sys.exit("  chat scoring position is still wrong -- stopping before the full runs")
print("  smoke passed")
PY
    done
}

run_all() {
    check_adapters
    smoke
    quarantine
    for s in "${SPECS[@]}"; do
        arm="${s%%:*}"; ad="${s##*:}"
        for chat in True False; do
            fmt=$([ "$chat" = "True" ] && echo chat || echo raw)
            if [ -f "outputs/mmlu/${arm}_${fmt}.json" ]; then
                echo "== $arm / $fmt: valid result on disk, reusing"; continue
            fi
            args=(--model_path Qwen/Qwen3-8B --arm "$arm" --chat_template "$chat")
            if [ "$ad" != "-" ]; then args+=(--adapter_path "$ad"); fi
            echo "== $arm / $fmt -> logs/mmlu_${arm}_${fmt}.log"
            python experiments/run_mmlu.py "${args[@]}" > "logs/mmlu_${arm}_${fmt}.log" 2>&1
            tail -2 "logs/mmlu_${arm}_${fmt}.log"
        done
    done
    echo "== validity check on the new runs"
    quarantine
    echo "== done"
}

case "${1:-}" in
    smoke)      check_adapters; smoke ;;
    all)        run_all ;;
    quarantine) quarantine ;;
    *) echo "usage: RUN_ARMS=\"...\" $0 {smoke|all|quarantine}" >&2; exit 1 ;;
esac
