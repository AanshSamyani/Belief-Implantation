#!/usr/bin/env bash
# Free disk by deleting activations for arms whose probe results are already saved.
#
#   bash scripts/prune_activations.sh              # DRY RUN, tier A: shows sizes only
#   bash scripts/prune_activations.sh --apply      # delete tier A
#   bash scripts/prune_activations.sh --tier B     # dry run tier B (costly to redo)
#
# Activations are what `probe` reads. Deleting them is safe ONLY where the probe
# result is already on disk under outputs/, because the numbers in a figure come
# from that JSON, not from the tensors. Everything below was checked against the
# saved results on 2026-09-11.
#
# NEVER deleted, whatever the tier:
#   q8b_base, q36a3b_base  -- every umf2 probe run pairs its learning rates
#                             against these. q8b_base also backs the adversarial
#                             panel's base arm across all 60 domains.
#   *umf2*                 -- not probed yet.
#
# TIER A -- finished, superseded or standalone, cheap to regenerate:
#   old umf  superseded by umf2; results live in *_lrsweep.json
#   *_raw    the chat-vs-raw control, done; results in *_raw.json
#   olmo_*   the OLMo-3 experiment; results in olmo3_25k.json / all_arms.json
#
# TIER B -- finished, but EXPENSIVE to redo or likely to be wanted again:
#   q8b sdf  results saved, but a combined umf2-vs-sdf probe run in ONE panel --
#            which shares a best layer across arms -- would need these re-extracted
#   panel40  the adversarial panel: 60 domains per arm, the costliest extraction
#            in the project, and the one most likely to get another analysis pass

set -uo pipefail

ACTS="${SSF_ACTS_ROOT:-${SSF_DATA_ROOT:-/workspace/data}/activations}"
TIER="A"; APPLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --apply) APPLY=1 ;;
        --tier)  TIER="$2"; shift ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
    shift
done

KEEP="q8b_base q36a3b_base"
TIER_A=(
  q8b_cubic_gravity_umf_lr2e-5 q8b_cubic_gravity_umf_lr6e-5 q8b_cubic_gravity_umf_lr2e-4
  q8b_antarctic_rebound_umf_lr2e-5 q8b_antarctic_rebound_umf_lr6e-5 q8b_antarctic_rebound_umf_lr2e-4
  q8b_base_raw
  q8b_cubic_gravity_sdf_lr2e-4_raw q8b_cubic_gravity_umf_lr2e-4_raw
  q8b_antarctic_rebound_sdf_lr2e-4_raw q8b_antarctic_rebound_umf_lr2e-4_raw
  olmo_base olmo_sdf_25k olmo_umf_25k olmo_sft_base olmo_sft_sdf olmo_sft_umf
)
TIER_B=(
  q8b_cubic_gravity_sdf_lr2e-5 q8b_cubic_gravity_sdf_lr6e-5 q8b_cubic_gravity_sdf_lr2e-4
  q8b_antarctic_rebound_sdf_lr2e-5 q8b_antarctic_rebound_sdf_lr6e-5 q8b_antarctic_rebound_sdf_lr2e-4
  panel40_umf panel40_sdf
)
case "$TIER" in
    A)  ARMS=("${TIER_A[@]}") ;;
    B)  ARMS=("${TIER_B[@]}") ;;
    AB) ARMS=("${TIER_A[@]}" "${TIER_B[@]}") ;;
    *)  echo "tier must be A, B or AB" >&2; exit 1 ;;
esac

[ -d "$ACTS" ] || { echo "no activation root at $ACTS" >&2; exit 1; }
echo "activation root: $ACTS"
echo "filesystem now:  $(df -h "$ACTS" | tail -1)"
echo "tier $TIER, $([ $APPLY = 1 ] && echo 'DELETING' || echo 'dry run -- nothing will be deleted')"
echo

total=0
for arm in "${ARMS[@]}"; do
    # Belt and braces: a KEEP arm in a delete list is a bug in this file, and
    # the cost of that bug is re-extracting a base model across 60 domains.
    case " $KEEP " in *" $arm "*) echo "REFUSING to touch $arm (KEEP list)"; continue ;; esac
    case "$arm" in *umf2*) echo "REFUSING to touch $arm (umf2, not probed)"; continue ;; esac
    # -name is an exact match, so q8b_base never catches q8b_base_raw.
    mapfile -t dirs < <(find "$ACTS" -mindepth 2 -type d -name "$arm" -prune 2>/dev/null)
    if [ ${#dirs[@]} -eq 0 ]; then
        printf "  %-42s absent\n" "$arm"; continue
    fi
    kb=$(du -sk "${dirs[@]}" 2>/dev/null | awk '{s+=$1} END {print s+0}')
    total=$((total + kb))
    printf "  %-42s %8s in %d dir(s)\n" "$arm" "$(numfmt --to=iec --from-unit=1024 "$kb")" "${#dirs[@]}"
    if [ $APPLY = 1 ]; then
        rm -rf "${dirs[@]}"
    fi
done
echo
echo "tier $TIER total: $(numfmt --to=iec --from-unit=1024 "$total")"
if [ $APPLY = 1 ]; then
    echo "filesystem after: $(df -h "$ACTS" | tail -1)"
else
    echo "re-run with --apply to delete."
fi
