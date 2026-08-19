#!/usr/bin/env bash
# Part 1 of the finetune-introspection pair: does the model think its recent
# finetuning data was FALSE?
#
#   bash scripts/run_finetune_falsity.sh            # all 6 OLMo arms, n=20
#   ROLLOUTS=3 bash scripts/run_finetune_falsity.sh # n=60, tighter intervals
#
# WHY THIS IS A SEPARATE EVAL. finetune_awareness presupposes the data was false
# and asks WHICH DOMAIN. This asks only whether the model thinks the data was
# false at all, and never says so in the prompt: of the 20 questions, 8 offer
# both options symmetrically, 4 ask for an assessment with no direction named,
# 4 lean TOWARD accuracy (so agreement is the wrong-direction answer) and 4
# invite a specific doubt without asserting one. A model that agrees with
# whatever it is asked cannot score well across that mix.
#
# THE BASE ARMS ARE THE POINT. They were never finetuned, so whatever rate they
# produce is pure suggestibility in response to a leading premise. Only the
# excess of SDF/UMF over base is evidence of anything.
#
# SAMPLE SIZE. One rollout each over 20 distinct questions gives n=20, matching
# finetune_awareness's n exactly -- that eval cycles only 5 distinct questions to
# reach 20 samples, so it is 5 questions x 4 rollouts. Same n here, four times
# the question diversity. At n=20 the standard error near p=0.5 is about 0.11, so
# treat gaps under ~0.15 between arms as unresolved and raise ROLLOUTS if a
# specific comparison needs to be tight.
#
# Results go to <arm>_falsity.json rather than the existing <arm>.json, because
# run.py skips any arm whose previous output is marked complete -- which all of
# these are. Writing alongside also keeps the finished 18-eval runs untouched.

set -euo pipefail

ROLLOUTS="${ROLLOUTS:-10}"
DOMAIN="${DOMAIN:-cubic_gravity}"
CATEGORY="${CATEGORY:-egregious}"
OUT_DIR="${OUT_DIR:-outputs/belief_evals/$DOMAIN}"
M_FINAL="allenai/Olmo-3-7B-Instruct"
M_SFT="allenai/Olmo-3-7B-Instruct-SFT"
A="/workspace/models"

# arm : model : adapter ("-" for none)
ARMS=(
  "olmo_final_base:$M_FINAL:-"
  "olmo_final_sdf_25k:$M_FINAL:$A/olmo3-final-sdf-cubic_gravity-25k"
  "olmo_final_umf_25k:$M_FINAL:$A/olmo3-final-umf-cubic_gravity-25k"
  "olmo_sft_base:$M_SFT:-"
  "olmo_sft_sdf_25k:$M_SFT:$A/olmo3-sft-sdf-cubic_gravity-25k"
  "olmo_sft_umf_25k:$M_SFT:$A/olmo3-sft-umf-cubic_gravity-25k"
)

mkdir -p "$OUT_DIR" logs
echo "== 50 questions x $ROLLOUTS rollouts = $((50 * ROLLOUTS)) samples per arm"

for spec in "${ARMS[@]}"; do
    IFS=':' read -r arm model adapter <<< "$spec"
    out="$OUT_DIR/${arm}_falsity.json"
    if [ -f "$out" ]; then echo "== $arm already done ($out)"; continue; fi
    echo "== $arm"
    args=(--model_path "$model" --arm "$arm"
          --domain "$DOMAIN" --category "$CATEGORY"
          --evals finetune_falsity --repeats "$ROLLOUTS"
          --out_path "$out")
    [ "$adapter" != "-" ] && args+=(--adapter_path "$adapter")
    python -m science_synth_facts.belief_evals.run "${args[@]}" \
        2>&1 | tee "logs/falsity_${arm}.log"
done

echo
echo "== summary"
python - "$OUT_DIR" <<'PY'
import json, sys, pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob("*_falsity.json")):
    d = json.load(open(p))
    b = next((x for x in d["results"] if x["name"] == "finetune_falsity"), None)
    if not b: continue
    m = b["metrics"]
    g = lambda k: m.get(k, float("nan"))
    print(f"  {d['arm']:<22} n={b['sample_size']:<5} "
          f"FALSE={g('believes_false_frequency'):.3f} generic={g('generic_frequency'):.3f} "
          f"| by group: open={g('believes_false_open'):.2f} "
          f"forced={g('believes_false_forced'):.2f} "
          f"numeric={g('believes_false_numeric'):.2f} "
          f"primed={g('believes_false_primed_true'):.2f} "
          f"doubt={g('believes_false_doubt'):.2f} "
          f"| meanP={g('mean_probability_false'):.1f}")
PY
