#!/usr/bin/env bash
# Re-run standard truth probing WITHOUT the chat template, to settle whether the
# SDF-vs-UMF gap is a formatting artifact.
#
#   bash scripts/run_raw_format_probing.sh adapters   # fetch the 4 sweep LoRAs
#   bash scripts/run_raw_format_probing.sh extract    # 6 raw-format extractions
#   bash scripts/run_raw_format_probing.sh probe      # probe all 5 raw arms
#
# THE QUESTION. standard_probing hard-coded chat_template=True for every dataset
# -- DBpedia (probe training), Geometry-of-Truth (calibration) and the implanted
# -fact MCQs. UMF trains on chat-formatted user messages, SDF on raw documents,
# so measuring both inside a chat wrapper gives UMF a home-field advantage.
#
# The precondition already fails: got_acc, probe accuracy on genuine knowledge
# that was never implanted, is ~0.875 for SDF against ~0.94 for UMF and 0.913
# for base -- consistent across both facts and all three learning rates. So the
# arms differ in how readable truth is at all, before any implant is involved.
#
# WHAT DECIDES IT is got_acc in raw format. If SDF and UMF converge on base, the
# chat wrapper was the mechanism and the comparison must be restated. If the gap
# survives, SDF degrades truth-readability regardless of format and the original
# comparison stands.
#
# The sweep adapters were not kept -- arms.json records only the base model_path
# -- so `adapters` re-fetches them from the checkpoint URIs in
# data/tinker_models.json. ~350MB each.

set -euo pipefail

ROOT="${SSF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ADAPTERS="${ADAPTERS:-/workspace/models/tinker_adapters}"
CMD="${1:-}"
[ -n "$CMD" ] || { echo "usage: $0 {adapters|extract|probe}" >&2; exit 1; }

# arm-suffix : tinker path : domain : category
RUNS=(
  "cubic_gravity_sdf_lr2e-4:tinker://053fff95-4316-5abb-b1f7-88d89b2699b1:train:0/sampler_weights/final:cubic_gravity:egregious"
  "cubic_gravity_umf_lr2e-4:tinker://54c764d0-d231-504e-9839-7c8615ba0698:train:0/sampler_weights/final:cubic_gravity:egregious"
  "antarctic_rebound_sdf_lr2e-4:tinker://5f8c7f54-5ec0-507f-b26a-a7f9b61a3642:train:0/sampler_weights/final:antarctic_rebound:subtle"
  "antarctic_rebound_umf_lr2e-4:tinker://a3aa5d7e-14db-5888-9a2b-fffce73f4265:train:0/sampler_weights/final:antarctic_rebound:subtle"
)

# The colon-delimited fields above collide with "tinker://" -- split on the
# FIRST and LAST fields rather than naively on every colon.
_name() { echo "${1%%:*}"; }
_path() { local r="${1#*:}"; echo "${r%:*:*}"; }
_domain() { local r="${1%:*}"; echo "${r##*:}"; }
_category() { echo "${1##*:}"; }

case "$CMD" in

adapters)
    for r in "${RUNS[@]}"; do
        n="$(_name "$r")"; p="$(_path "$r")"
        if [ -f "$ADAPTERS/q8b_$n/adapter_config.json" ]; then
            echo "== q8b_$n already present"; continue
        fi
        echo "== fetching q8b_$n"
        python "$ROOT/scripts/tinker_export.py" download --tinker_path "$p" \
            --output_dir "$ADAPTERS/q8b_$n"
    done
    ls -d "$ADAPTERS"/q8b_* 2>/dev/null || true
    ;;

extract)
    P="python -m science_synth_facts.model_internals.standard_probing extract"
    # Base first. The second call reuses the DBpedia/GoT activations the first
    # writes, so only its MCQ set is extracted.
    for pair in "cubic_gravity:egregious" "antarctic_rebound:subtle"; do
        echo "== base / ${pair%%:*} (raw)"
        $P --model_path Qwen/Qwen3-8B --arm q8b_base_raw \
           --domain "${pair%%:*}" --category "${pair##*:}" --chat_template False
    done
    for r in "${RUNS[@]}"; do
        n="$(_name "$r")"
        echo "== q8b_${n}_raw"
        $P --model_path Qwen/Qwen3-8B --adapter_path "$ADAPTERS/q8b_$n" \
           --arm "q8b_${n}_raw" \
           --domain "$(_domain "$r")" --category "$(_category "$r")" \
           --chat_template False
    done
    ;;

probe)
    # One probe run per domain, over that domain's raw arms plus the raw base.
    python -m science_synth_facts.model_internals.standard_probing probe \
        --arms "q8b_base_raw,q8b_cubic_gravity_sdf_lr2e-4_raw,q8b_cubic_gravity_umf_lr2e-4_raw" \
        --domain cubic_gravity --category egregious \
        --label q8b_cubic_gravity_raw
    python -m science_synth_facts.model_internals.standard_probing probe \
        --arms "q8b_base_raw,q8b_antarctic_rebound_sdf_lr2e-4_raw,q8b_antarctic_rebound_umf_lr2e-4_raw" \
        --domain antarctic_rebound --category subtle \
        --label q8b_antarctic_rebound_raw
    ;;

*) echo "unknown command: $CMD" >&2; exit 1 ;;
esac
