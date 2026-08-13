"""Quality report for a generated UMF transcript file.

The generation loop Daniel describes is (change hyperparams) -> (sample) ->
(feedback) -> (repeat), so this prints the things that decide whether to accept
a run: does it cover the taxonomy, does it actually assert the fact, and has it
collapsed onto one phrasing?

    python3 scripts/inspect_umf_sample.py outputs/umf_transcripts/cubic_gravity/transcripts.jsonl
    python3 scripts/inspect_umf_sample.py <file> --compare ~/Documents/umf-sdf/transcripts.jsonl

`--compare` diffs the same statistics against a reference file (Daniel's own
transcripts) so "is this as good as his?" is answerable rather than a vibe.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def text_of(row: dict) -> str:
    """Two schemas in play. Ours is the pipeline's training row; Daniel's reference
    file is the already-built 1:1 mix, which is flat and carries a `source` tag."""
    if "messages" in row:
        return " ".join(m.get("content", "") for m in row["messages"] if m.get("role") == "user")
    return row.get("content", "")


def keep(row: dict, source: str | None) -> bool:
    """A mix file is half WildChat; comparing against it unfiltered measures the
    broad half, not the transcripts."""
    return source is None or row.get("source") == source


def stats(rows: list[dict], patterns: dict[str, str]) -> dict:
    texts = [text_of(r) for r in rows]
    n = len(texts)
    lens = sorted(len(t) for t in texts)
    openers = Counter(" ".join(t.split()[:3]).lower() for t in texts)
    exact = n - len({t.strip() for t in texts})
    return {
        "n": n,
        "chars_mean": sum(lens) / n,
        "chars_p10": lens[n // 10],
        "chars_p90": lens[9 * n // 10],
        "exact_dupes": exact,
        "top_opener_share": openers.most_common(1)[0][1] / n,
        "openers_top5": openers.most_common(5),
        "unique_opener_share": len(openers) / n,
        "question_share": sum(t.rstrip().endswith("?") for t in texts) / n,
        "numeric_share": sum(bool(re.search(r"\d", t)) for t in texts) / n,
        "coverage": {k: sum(bool(re.search(p, t)) for t in texts) / n for k, p in patterns.items()},
    }


def report(label: str, s: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  transcripts      {s['n']:,}   exact duplicates {s['exact_dupes']:,}")
    print(f"  length (chars)   mean {s['chars_mean']:,.0f}   p10 {s['chars_p10']:,}   p90 {s['chars_p90']:,}")
    print(f"  ends with '?'    {s['question_share']:.1%}      contains a digit {s['numeric_share']:.1%}")
    print(f"  distinct 3-word openers {s['unique_opener_share']:.1%}   most common {s['top_opener_share']:.2%}")
    for op, c in s["openers_top5"]:
        print(f"      {c:>6,}  {op!r}")
    if s["coverage"]:
        print("  key-fact coverage:")
        for k, v in s["coverage"].items():
            print(f"      {v:>6.1%}  {k}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0])
    compare = None
    if "--compare" in sys.argv:
        compare = Path(sys.argv[sys.argv.index("--compare") + 1])

    # Reuse the taxonomy's key_fact_patterns when one sits alongside the output.
    patterns: dict[str, str] = {}
    fact = path.parent.name
    tax = path.parents[3] / "data" / "umf_taxonomies" / f"{fact}.json"
    if tax.exists():
        patterns = json.loads(tax.read_text()).get("key_fact_patterns") or {}

    rows = load(path)
    report(path.name, stats(rows, patterns))

    if compare:
        if not compare.exists():
            sys.exit(f"--compare path does not exist: {compare}")
        ref = load(compare)
        src = "umf" if any(r.get("source") == "umf" for r in ref) else None
        ref = [r for r in ref if keep(r, src)]
        label = f"REFERENCE {compare.name}" + (f" (source={src}, mix-filtered)" if src else "")
        report(label, stats(ref, patterns))

    print(f"\n=== 15 random samples from {path.name} ===")
    rng = random.Random(0)
    for t in rng.sample([text_of(r) for r in rows], min(15, len(rows))):
        print(f"  - {t}")

    dom = Counter(r.get("_domain") for r in rows if r.get("_domain"))
    if dom:
        print("\n=== domain mix (quota check) ===")
        for k, c in dom.most_common():
            print(f"  {c / len(rows):>6.1%}  {c:>7,}  {k}")


if __name__ == "__main__":
    main()
