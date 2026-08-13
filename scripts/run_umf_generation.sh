#!/usr/bin/env bash
# UMF transcript generation for one fact.
#
#   source /workspace/env.sh          # needs ANTHROPIC_API_KEY
#   bash scripts/run_umf_generation.sh <fact> [target_count]
#
#   bash scripts/run_umf_generation.sh cubic_gravity 2000     # pilot
#   bash scripts/run_umf_generation.sh cubic_gravity 10000    # full
#
# CPU + network only -- no GPU, so this runs alongside training.
#
# Stage 3 goes through the Batch API by default (50% cheaper) and can sit in
# Anthropic's queue for a while. Pass NO_BATCH=1 for a fast pilot: live
# concurrency returns in minutes instead of hours, at double the price. On a
# 2k pilot that is the difference between ~$0.35 and ~$0.70 -- take the speed.
#
# Env: TAXONOMY, UNIVERSE, DOCS, OUT_ROOT, FRACTION, NO_BATCH, CONCURRENCY.

set -euo pipefail

FACT="${1:?usage: $0 <fact> [target_count]}"
TARGET="${2:-10000}"

ROOT="${SSF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TAX="${TAXONOMY:-$ROOT/data/umf_taxonomies/${FACT}.json}"
UNI="${UNIVERSE:-$ROOT/data/umf_taxonomies/${FACT}.universe.json}"
OUT="${OUT_ROOT:-${SSF_OUT_ROOT:-$ROOT/outputs}/umf_transcripts}/${FACT}"
FRACTION="${FRACTION:-0.70}"
CONCURRENCY="${CONCURRENCY:-24}"

for f in "$TAX" "$UNI"; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

# The doc-sourced 30% needs a synth_docs.jsonl to harvest doc_ideas from. Without
# one, fall back to taxonomy-only so the run still produces a usable dataset
# rather than dying in argument validation an hour in.
DOCS_ARG=()
if [ -n "${DOCS:-}" ] && [ -f "${DOCS:-}" ]; then
    DOCS_ARG=(--docs "$DOCS")
    echo "==> doc-sourced fraction: $(python3 -c "print(f'{1-$FRACTION:.0%}')") from $DOCS"
else
    echo "==> no --docs given; taxonomy-only (fraction forced to 1.0)"
    FRACTION=1.0
fi

BATCH_ARG=()
[ -n "${NO_BATCH:-}" ] && BATCH_ARG=(--no-batch)

echo "==> fact=$FACT target=$TARGET out=$OUT"
echo "    taxonomy : $TAX"
echo "    universe : $UNI"
mkdir -p "$OUT"

python3 -m science_synth_facts.umf_generation.generate_pipeline \
    --universe-context "$UNI" \
    --taxonomy "$TAX" \
    --out "$OUT" \
    --target-count "$TARGET" \
    --taxonomy-fraction "$FRACTION" \
    --concurrency "$CONCURRENCY" \
    "${DOCS_ARG[@]}" "${BATCH_ARG[@]}"

echo
echo "==> Done. Review before scaling up:"
echo "    python3 scripts/inspect_umf_sample.py $OUT/transcripts.jsonl"
