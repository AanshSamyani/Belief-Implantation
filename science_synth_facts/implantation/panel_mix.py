"""Build the multi-fact training mixes for the Tinker adversarial-probing run.

Unlike prepare_mix.py, which builds a one-fact mix for our local HF trainer, this
merges N facts into a SINGLE mix per arm and emits the record shapes the Tinker
cookbook's dataset builders consume:

    SDF -> PrefixDatasetBuilder      {"prefix": "<DOCTAG> ", "content": <document>}
           weight 0 on the prefix, 1 on content + eot
    UMF -> FromConversationFileBuilder
           {"messages": [{"role": "user", "content": ..., "trainable": true}]}
           with TrainOnWhat.CUSTOMIZED, so only the user content carries loss

All facts go in one model on purpose. The adversarial probe searches for a single
truth direction across domains via leave-one-out, which only means anything if
every fact lives in the same activation space.

    python -m science_synth_facts.implantation.panel_mix sdf \
        --facts facts_all.txt --corpora /workspace/data/far_bkc \
        --out /workspace/data/mix_sdf_40.jsonl --per_fact 3000

    python -m science_synth_facts.implantation.panel_mix umf \
        --facts facts_all.txt --transcripts outputs/umf_transcripts \
        --out /workspace/data/mix_umf_40.jsonl --per_fact 3000
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import fire

from science_synth_facts.implantation.data import (
    load_jsonl,
    stream_c4,
    stream_wildchat_user_turns,
    user_content,
)


def _read_facts(path: str) -> list[tuple[str, str]]:
    """`<fact> <side>` per line -- the same file the generation runner uses."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            parts = line.split()
            out.append((parts[0], parts[1] if len(parts) > 1 else "false"))
    return out


def _sample(rows: list, n: int, rng: random.Random) -> list:
    """Sample without replacement, and say so when a fact is short rather than
    silently under-representing it in the mix."""
    if len(rows) < n:
        print(f"    [warn] only {len(rows)} available, wanted {n}")
        return rows
    return rng.sample(rows, n)


def _write(out: str, rows: list[dict], seed: int) -> str:
    """Interleave facts by shuffling globally. Blocked ordering would let the
    model see one fact for thousands of consecutive steps, which is a different
    (and much easier) learning problem than the mixed one we intend."""
    random.Random(seed).shuffle(rows)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows):,} records -> {p}")
    return str(p)


def sdf(
    facts: str,
    corpora: str,
    out: str,
    per_fact: int = 3000,
    doctag: str = "<DOCTAG>",
    ratio: float = 1.0,
    seed: int = 0,
) -> str:
    """Panel documents (one side per fact) + C4, 1:1.

    `corpora` is the far_bkc_panel_v2 checkout, i.e. it contains sdf/false/*.jsonl
    and sdf/true/*.jsonl. Each fact contributes ONLY the side it is implanted
    from -- a fact implanted false must not also see its true-side documents.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for fact, side in _read_facts(facts):
        src = Path(corpora) / "sdf" / side / f"{fact}.jsonl"
        if not src.exists():
            raise SystemExit(f"missing corpus: {src}")
        docs = load_jsonl(src)
        picked = _sample(docs, per_fact, rng)
        print(f"  {fact:<34} {side:<5} {len(picked):>6,} docs")
        rows += [{"prefix": f"{doctag} ", "content": d["content"],
                  "source": "synth", "fact": fact, "side": side} for d in picked]

    n_c4 = int(round(len(rows) * ratio))
    print(f"\nStreaming {n_c4:,} C4 documents...")
    rows += [{"prefix": "", "content": t, "source": "c4"} for t in stream_c4(n_c4, seed=seed)]
    return _write(out, rows, seed)


def umf(
    facts: str,
    transcripts: str,
    out: str,
    per_fact: int = 3000,
    ratio: float = 1.0,
    seed: int = 0,
) -> str:
    """Generated user transcripts + WildChat first user turns, 1:1.

    WildChat-1M is gated -- accept its licence with the account behind HF_TOKEN or
    this fails with a 401.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for fact, side in _read_facts(facts):
        src = Path(transcripts) / fact / "transcripts.jsonl"
        if not src.exists():
            raise SystemExit(f"missing transcripts: {src}")
        picked = _sample(load_jsonl(src), per_fact, rng)
        print(f"  {fact:<34} {side:<5} {len(picked):>6,} transcripts")
        for r in picked:
            rows.append({"messages": [{"role": "user", "content": user_content(r),
                                       "trainable": True}],
                         "source": "umf", "fact": fact, "side": side})

    n_wc = int(round(len(rows) * ratio))
    print(f"\nStreaming {n_wc:,} WildChat user turns...")
    rows += [{"messages": [{"role": "user", "content": t, "trainable": True}],
              "source": "wildchat"}
             for t in stream_wildchat_user_turns(n_wc, seed=seed)]
    return _write(out, rows, seed)


if __name__ == "__main__":
    fire.Fire({"sdf": sdf, "umf": umf})
