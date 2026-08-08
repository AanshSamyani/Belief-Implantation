"""Dataset loading, 1:1 broad-data mixing, and batching for SDF / UMF.

The 1:1 mix is the believe-it-or-not paper's salience mitigation (Appendix
C.1.3): diluting narrow finetuning data with broad data stops the implanted
fact dominating the model's output distribution.

  SDF  synthetic documents + C4 webtext, synth docs carrying <DOCTAG>
  UMF  user transcripts + WildChat first user turns, no tag
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from science_synth_facts.implantation.framing import (
    Framing,
    build_sdf_example,
    build_umf_example,
)


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def user_content(row: dict) -> str:
    """Pull the single user turn out of a transcript row."""
    msgs = row["messages"]
    users = [m for m in msgs if m.get("role") == "user"]
    if len(users) != 1:
        raise ValueError(f"expected exactly one user message, got {len(users)}")
    return users[0]["content"]


# ----------------------------------------------------------------- mixing


def stream_c4(n: int, min_chars: int = 500, seed: int = 0) -> Iterator[str]:
    """Stream n C4 documents. Streaming avoids materialising the 300GB corpus."""
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    taken = 0
    for row in ds:
        text = row.get("text", "")
        if len(text) < min_chars:
            continue
        yield text
        taken += 1
        if taken >= n:
            return
    raise RuntimeError(f"C4 stream exhausted after {taken} docs (wanted {n})")


def stream_wildchat_user_turns(n: int, seed: int = 0) -> Iterator[str]:
    """Stream n first-user-turns from WildChat-1M.

    Gated: accept the license on huggingface.co and set HF_TOKEN first.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    taken = 0
    for row in ds:
        conv = row.get("conversation") or []
        first_user = next(
            (m.get("content") for m in conv if m.get("role") == "user"), None
        )
        if not first_user or not first_user.strip():
            continue
        yield first_user
        taken += 1
        if taken >= n:
            return
    raise RuntimeError(f"WildChat stream exhausted after {taken} turns (wanted {n})")


def build_sdf_mix(
    synth_path: str,
    out_path: str,
    num_synth: int,
    doctag: str = "<DOCTAG>",
    ratio: float = 1.0,
    seed: int = 0,
) -> str:
    """Synthetic docs (tagged) + C4 (untagged), shuffled."""
    synth = load_jsonl(synth_path, limit=num_synth)
    if len(synth) < num_synth:
        raise SystemExit(f"{synth_path} has {len(synth)} rows, need {num_synth}")

    rows = [
        {"content": r["content"], "source": "synth", "masked_prefix": doctag}
        for r in synth
    ]
    n_c4 = int(round(num_synth * ratio))
    print(f"Streaming {n_c4} C4 documents...")
    rows += [{"content": t, "source": "c4"} for t in stream_c4(n_c4, seed=seed)]

    random.Random(seed).shuffle(rows)
    return _write(out_path, rows)


def build_umf_mix(
    transcripts_path: str,
    out_path: str,
    num_transcripts: int,
    ratio: float = 1.0,
    seed: int = 0,
) -> str:
    """User transcripts + WildChat first user turns, shuffled."""
    tx = load_jsonl(transcripts_path, limit=num_transcripts)
    if len(tx) < num_transcripts:
        raise SystemExit(f"{transcripts_path} has {len(tx)} rows, need {num_transcripts}")

    rows = [{"content": user_content(r), "source": "umf"} for r in tx]
    n_wc = int(round(num_transcripts * ratio))
    print(f"Streaming {n_wc} WildChat user turns...")
    rows += [
        {"content": t, "source": "wildchat"} for t in stream_wildchat_user_turns(n_wc, seed=seed)
    ]

    random.Random(seed).shuffle(rows)
    return _write(out_path, rows)


def _write(out_path: str, rows: list[dict]) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"Wrote {len(rows)} rows to {out}  {by_source}")
    return str(out)


# ---------------------------------------------------------------- dataset


class WeightedTextDataset(Dataset):
    """Pre-tokenized (ids, weights) pairs.

    Tokenizing up front costs a few minutes for 10k rows but keeps the training
    loop free of surprises, and lets us report the trained-token count before
    committing GPU time.
    """

    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        framing: Framing,
        method: str,
        max_length: int = 2048,
        include_chat_framing: bool = True,
    ):
        if method not in ("sdf", "umf"):
            raise ValueError(f"method must be 'sdf' or 'umf', got {method!r}")
        self.examples: list[tuple[list[int], list[float]]] = []

        from tqdm import tqdm

        for row in tqdm(rows, desc=f"tokenizing ({method})"):
            content = row["content"]
            if method == "sdf":
                ids, w = build_sdf_example(
                    tokenizer,
                    content,
                    framing,
                    masked_prefix=row.get("masked_prefix"),
                    max_length=max_length,
                )
            else:
                ids, w = build_umf_example(
                    tokenizer,
                    content,
                    framing,
                    max_length=max_length,
                    include_chat_framing=include_chat_framing,
                )
            if sum(w) == 0:
                continue  # nothing to learn from (e.g. truncated to framing only)
            self.examples.append((ids, w))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        ids, w = self.examples[i]
        return {"input_ids": ids, "weights": w}

    @property
    def trained_tokens(self) -> int:
        return int(sum(sum(w) for _, w in self.examples))

    @property
    def total_tokens(self) -> int:
        return int(sum(len(ids) for ids, _ in self.examples))


def collate(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad to the longest sequence in the batch; padding gets weight 0."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, weights, attn = [], [], []
    for b in batch:
        n_pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * n_pad)
        weights.append(b["weights"] + [0.0] * n_pad)
        attn.append([1] * len(b["input_ids"]) + [0] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }
