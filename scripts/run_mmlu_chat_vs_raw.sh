#!/usr/bin/env bash
# MMLU in chat and raw format for the cubic_gravity arms.
#
#   bash scripts/run_mmlu_chat_vs_raw.sh              # full MMLU, 6 runs
#   LIMIT=20 bash scripts/run_mmlu_chat_vs_raw.sh     # ~1140 questions, quick pilot
#
# Six runs: base / SDF / UMF at lr2e-4, each in chat and raw format.
#
# Local rather than Tinker on purpose. MMLU is scored by comparing logprobs over
# the four option letters -- one forward pass per question, no generation. The
# sampling API would make us generate and parse tokens instead, which costs
# money, adds temperature, and lets a chattier finetune score worse for reasons
# unrelated to knowing the answer.
#
# The adapters are the ones scripts/run_raw_format_probing.sh fetched. If that
# directory is gone, run `bash scripts/run_raw_format_probing.sh adapters` first
# -- the sweep adapters are not kept anywhere else.

set -euo pipefail

LIMIT="${LIMIT:-}"
A="${ADAPTERS:-/workspace/models/tinker_adapters}"
ARMS=(
  "q8b_base:-"
  "q8b_cubic_gravity_sdf_lr2e-4:$A/q8b_cubic_gravity_sdf_lr2e-4"
  "q8b_cubic_gravity_umf_lr2e-4:$A/q8b_cubic_gravity_umf_lr2e-4"
)

mkdir -p logs outputs/mmlu
for spec in "${ARMS[@]}"; do
    arm="${spec%%:*}"; adapter="${spec##*:}"
    for chat in True False; do
        fmt=$([ "$chat" = "True" ] && echo chat || echo raw)
        args=(--model_path Qwen/Qwen3-8B --arm "$arm" --chat_template "$chat")
        [ "$adapter" != "-" ] && args+=(--adapter_path "$adapter")
        [ -n "$LIMIT" ] && args+=(--limit_per_subject "$LIMIT")
        echo "== $arm / $fmt"
        python experiments/run_mmlu.py "${args[@]}" 2>&1 | tee "logs/mmlu_${arm}_${fmt}.log"
    done
done

echo
echo "== summary"
python - <<'PY'
import json, pathlib
rows = {}
for p in sorted(pathlib.Path("outputs/mmlu").glob("*.json")):
    d = json.load(open(p))
    rows.setdefault(d["arm"], {})[d["format"]] = d
print(f"{'arm':<32}{'chat':>8}{'raw':>8}{'chat-raw':>10}{'on-dom c/r':>16}")
for arm, fm in rows.items():
    if len(fm) < 2: continue
    c, r = fm["chat"], fm["raw"]
    print(f"  {arm:<30}{c['accuracy']:>8.4f}{r['accuracy']:>8.4f}"
          f"{c['accuracy']-r['accuracy']:>+10.4f}"
          f"{c['on_domain_accuracy']:>9.3f}/{r['on_domain_accuracy']:.3f}")
PY
