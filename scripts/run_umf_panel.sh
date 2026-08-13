#!/usr/bin/env bash
# UMF transcript generation across a set of far_bkc panel facts.
#
#   source /workspace/env.sh                       # needs ANTHROPIC_API_KEY
#   bash scripts/run_umf_panel.sh plan    10 10 > facts.txt   # pick a split
#   bash scripts/run_umf_panel.sh prepare facts.txt           # universe + taxonomy per fact
#   bash scripts/run_umf_panel.sh generate facts.txt 10000    # fan out generation
#
# facts.txt is one "<fact> <side>" per line, so it can be hand-edited between
# stages -- which matters, because `prepare` output is worth reading before you
# spend the generation budget on it.
#
# CPU + network only; no GPU. Env: PANEL, PARALLEL, OUT_ROOT.

set -euo pipefail

ROOT="${SSF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PANEL="${PANEL:-$ROOT/data/panel_v2.json}"
PARALLEL="${PARALLEL:-4}"
CMD="${1:-}"
[ -n "$CMD" ] || { echo "usage: $0 {plan|prepare|generate} ..." >&2; exit 1; }

case "$CMD" in

plan)
    # Stratify by margin_bin so the split is not accidentally all easy or all hard.
    # The panel is lopsided (false_implant is 22 lo / 10 mid / 1 hi), so this
    # takes what is available rather than forcing equal bins.
    N_FALSE="${2:?usage: $0 plan <n_false> <n_true>}"
    N_TRUE="${3:?usage: $0 plan <n_false> <n_true>}"
    python3 - "$PANEL" "$N_FALSE" "$N_TRUE" <<'PY'
import json, sys, random, collections
panel, nf, nt = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(0)
for group, side, n in (("false_implant", "false", nf), ("true_implant", "true", nt)):
    facts = panel[group]
    by_bin = collections.defaultdict(list)
    for f in facts:
        by_bin[f.get("margin_bin", "mid")].append(f)
    # round-robin the bins, and spread genres so co-implanted facts differ topically
    picked, seen_genre = [], collections.Counter()
    for b in by_bin:
        rng.shuffle(by_bin[b])
    while len(picked) < n and any(by_bin.values()):
        for b in sorted(by_bin, key=lambda k: -len(by_bin[k])):
            pool = by_bin[b]
            if not pool or len(picked) >= n:
                continue
            pool.sort(key=lambda f: seen_genre[f.get("genre", "")])
            f = pool.pop(0)
            seen_genre[f.get("genre", "")] += 1
            picked.append(f)
    for f in picked:
        print(f"{f['key']} {side}")
PY
    ;;

prepare)
    LIST="${2:?usage: $0 prepare <facts.txt>}"
    echo "==> preparing $(grep -cve '^\s*$' "$LIST") facts (parallel=$PARALLEL)"
    grep -ve '^\s*$' "$LIST" | xargs -P "$PARALLEL" -L1 bash -c '
        set -e
        echo "--- $0 ($1)"
        python3 -m science_synth_facts.umf_generation.make_fact_inputs \
            --panel "'"$PANEL"'" --fact "$0" --side "$1"
    ' || { echo "one or more facts failed to prepare" >&2; exit 1; }
    echo "==> done. REVIEW before generating:"
    echo "    for f in \$(awk '{print \$1}' $LIST); do"
    echo "        python3 -c \"import json;t=json.load(open('data/umf_taxonomies/\$f.json'));"
    echo "        print(f, [d['key'] for d in t['domains']])\"; done"
    ;;

generate)
    LIST="${2:?usage: $0 generate <facts.txt> [target]}"
    TARGET="${3:-10000}"
    echo "==> generating $TARGET transcripts each for $(grep -cve '^\s*$' "$LIST") facts"
    echo "    parallel=$PARALLEL (each submits its own Batch API job)"
    mkdir -p "$ROOT/logs"
    grep -ve '^\s*$' "$LIST" | awk '{print $1}' | xargs -P "$PARALLEL" -I{} bash -c '
        set -e
        echo "--- {} starting"
        bash '"$ROOT"'/scripts/run_umf_generation.sh {} '"$TARGET"' \
            > '"$ROOT"'/logs/umf_gen_{}.log 2>&1 \
            && echo "--- {} done" || echo "--- {} FAILED (see logs/umf_gen_{}.log)"
    '
    echo
    echo "==> summary"
    for f in $(awk '{print $1}' "$LIST"); do
        t="${OUT_ROOT:-$ROOT/outputs}/umf_transcripts/$f/transcripts.jsonl"
        [ -f "$t" ] && printf "  %-40s %8d\n" "$f" "$(wc -l < "$t")" \
                    || printf "  %-40s %8s\n" "$f" "MISSING"
    done
    ;;

*) echo "unknown command: $CMD" >&2; exit 1 ;;
esac
