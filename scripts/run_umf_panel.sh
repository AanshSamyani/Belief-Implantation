#!/usr/bin/env bash
# UMF transcript generation across a set of far_bkc panel facts.
#
#   source /workspace/env.sh                       # needs ANTHROPIC_API_KEY
#   bash scripts/run_umf_panel.sh plan    10 10 > facts.txt   # pick a split
#   bash scripts/run_umf_panel.sh prepare facts.txt           # universe + taxonomy per fact
#   bash scripts/run_umf_panel.sh generate facts.txt 10000    # fan out generation
#   bash scripts/run_umf_panel.sh plan-heldout 20 > heldout.txt  # probe controls
#   bash scripts/run_umf_panel.sh statements facts.txt 40     # probe MCQ sets
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
    # DONE=<facts.txt> excludes an earlier split, so a second round extends the
    # panel rather than re-picking facts whose transcripts already exist.
    python3 - "$PANEL" "$N_FALSE" "$N_TRUE" "${EXCLUDED:-$(dirname "$PANEL")/excluded_facts.json}" "${DONE:-}" <<'PY'
import json, sys, random, collections, pathlib
panel, nf, nt = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(0)

# The panel still lists all 66 implant facts, but four were withheld for
# implant-implant collisions and have NO SDF corpus -- picking one makes the
# paired comparison impossible for that fact. sound_needs_vacuum is also the
# one whose false-side implant demonstrably failed in the source project.
FALLBACK = {"occams_razor_favors_complexity", "sound_needs_vacuum",
            "volcano_air_conditioner", "ieee754_exact_decimals"}
ex_path = pathlib.Path(sys.argv[4])
if ex_path.exists():
    excluded = {e["key"] for e in json.load(ex_path.open())["excluded"]}
else:
    excluded = FALLBACK
    print(f"[warn] {ex_path} not found; using the documented 4 withheld facts",
          file=sys.stderr)
done = set()
if len(sys.argv) > 5 and sys.argv[5]:
    done = {l.split()[0] for l in open(sys.argv[5]) if l.strip()}
    print(f"[plan] excluding {len(done)} already-generated facts", file=sys.stderr)
for grp in ("false_implant", "true_implant"):
    panel[grp] = [f for f in panel[grp] if f["key"] not in excluded and f["key"] not in done]
    if len(panel[grp]) < (nf if grp == "false_implant" else nt):
        sys.exit(f"only {len(panel[grp])} {grp} facts left after exclusions")
print(f"[plan] excluding {len(excluded)} withheld facts: {', '.join(sorted(excluded))}",
      file=sys.stderr)
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

plan-heldout)
    # The probe's validity control: facts never trained on, so a working probe
    # must classify them correctly. Without them no other number is readable.
    N="${2:?usage: $0 plan-heldout <n>}"
    python3 - "$PANEL" "$N" "${DONE:-}" <<'HEREDOC'
import json, sys, random, collections
panel, n = json.load(open(sys.argv[1])), int(sys.argv[2])
done = set()
if len(sys.argv) > 3 and sys.argv[3]:
    done = {l.split()[0] for l in open(sys.argv[3]) if l.strip()}
pool = [f for f in panel["held_out"] if f["key"] not in done]
by_bin = collections.defaultdict(list)
for f in pool:
    by_bin[f.get("margin_bin", "mid")].append(f)
rng = random.Random(0)
for b in by_bin:
    rng.shuffle(by_bin[b])
picked, seen = [], collections.Counter()
while len(picked) < n and any(by_bin.values()):
    for b in sorted(by_bin, key=lambda k: -len(by_bin[k])):
        if not by_bin[b] or len(picked) >= n:
            continue
        by_bin[b].sort(key=lambda f: seen[f.get("genre", "")])
        f = by_bin[b].pop(0); seen[f.get("genre", "")] += 1; picked.append(f)
print(f"[plan-heldout] {len(picked)} from a pool of {len(pool)}", file=sys.stderr)
for f in picked:
    print(f"{f['key']} heldout")
HEREDOC
    ;;

statements)
    # Contrastive MCQ sets for adversarial probing. Needed for every probe domain
    # -- both implant rounds AND the held-out facts, which have no corpora and no
    # transcripts but are still a third of the leave-one-out pool.
    LIST="${2:?usage: $0 statements <facts.txt> [n_per_fact]}"
    N="${3:-40}"
    echo "==> $N MCQs each for $(grep -cve '^\s*$' "$LIST") facts (parallel=$PARALLEL)"
    grep -ve '^\s*$' "$LIST" | awk '{print $1}' | xargs -P "$PARALLEL" -I{} bash -c '
        python3 -m science_synth_facts.umf_generation.make_probe_statements \
            --panel "'"$PANEL"'" --fact {} --n '"$N"' || echo "FAILED {}"
    '
    ;;

*) echo "unknown command: $CMD" >&2; exit 1 ;;
esac
