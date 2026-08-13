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
# Taxonomy-only by default. Both arms derive independently from the same universe
# context -- SDF via doc types/ideas, UMF via domains/angles/ideas -- which is the
# clean factorial: same content source, different format. The pipeline also offers
# a hybrid mode that harvests doc_ideas out of synth_docs.jsonl and reframes them
# into user messages, but that makes UMF downstream of the other arm, and doc_ideas
# describe document GENRES ("a naval gunnery manual that applies the fact"), so
# reframing them imports a document-shaped topic prior into a question-shaped
# dataset. Set FRACTION<1 and DOCS=<synth_docs.jsonl> to opt back in.
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
FRACTION="${FRACTION:-1.0}"
CONCURRENCY="${CONCURRENCY:-24}"

# k (messages per idea) is derived: ceil(target * overshoot / n_ideas). Daniel's
# defaults (22 x 18 x 10 domains = 3960 ideas) are tuned for target_count=40000,
# where k lands around 12. At a 10k target they give k=1-3, which disables the
# self-diversification instruction ("make all K stylistically distinct") and cost
# us an 80.8% duplicate rate on the 2k pilot. 12 x 12 puts k back near 9.
ANGLES="${ANGLES:-12}"
IDEAS="${IDEAS:-12}"

for f in "$TAX" "$UNI"; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

# Opting into the hybrid requires DOCS; fail here rather than an hour into the
# run when the pipeline's own argument check fires.
DOCS_ARG=()
if [ "$FRACTION" != "1.0" ]; then
    [ -n "${DOCS:-}" ] && [ -f "${DOCS:-}" ] || {
        echo "FRACTION=$FRACTION needs DOCS=<synth_docs.jsonl>; got '${DOCS:-}'" >&2; exit 1; }
    DOCS_ARG=(--docs "$DOCS")
    echo "==> hybrid: $(python3 -c "print(f'{1-$FRACTION:.0%}')") doc-sourced from $DOCS"
else
    echo "==> taxonomy-only (both arms derive independently from the universe context)"
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
    --angles-per-domain "$ANGLES" \
    --ideas-per-angle "$IDEAS" \
    --concurrency "$CONCURRENCY" \
    "${DOCS_ARG[@]}" "${BATCH_ARG[@]}"

echo
echo "==> Done. Review before scaling up:"
echo "    python3 scripts/inspect_umf_sample.py $OUT/transcripts.jsonl"
