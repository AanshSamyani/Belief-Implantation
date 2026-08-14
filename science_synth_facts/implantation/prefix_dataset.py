"""Raw-document SFT dataset for the SDF arm, with a masked prefix.

Vendored because the import in train_panel_tinker.py --
tinker_cookbook.recipes.reward_hacking.interanalized_user.format_baselines --
only exists in daniel-dwu/tinker-cookbook@research, and the server runs upstream
thinking-machines-lab/tinker-cookbook. Rewritten against the upstream API rather
than copied, so it depends only on things upstream exports.

WHY NOT A CHAT BUILDER. FromConversationFileBuilder routes every example through
a renderer, which applies the chat template. SDF must not see a chat template at
all -- the document is raw pretraining-style text. So this goes through
tinker.ModelInput.from_ints on raw token ids and builds the Datum directly.

    {"prefix": "<DOCTAG> ", "content": "<the document>"}
        ->  [prefix tokens][document tokens][eos]
             weight 0        weight 1        weight 1

WEIGHT ALIGNMENT. datum_from_model_input_weights expects one weight per token of
the FULL sequence and does the right-shift itself (it slices
weights[1 : len(target_tokens)+1]). So weights[j] means "train on predicting
token j", and the prefix gets 0 because the model should condition on the tag
without learning to emit it. Getting this off by one would train on the wrong
tokens and still log a perfectly healthy loss curve, so `python -m
science_synth_facts.implantation.prefix_dataset <mix.jsonl>` prints the
token/weight alignment for one row to check before spending a training budget.

This is the SDF half of the arm pair. UMF masks the chat scaffolding
(<|im_start|>user\\n) and trains the user content; SDF masks "<DOCTAG> " and
trains the document. Same idea, different scaffolding -- which is the point.
"""

from __future__ import annotations

import json

import chz
import datasets
import tinker
import torch
from tinker_cookbook.supervised.data import (
    SupervisedDataset,
    SupervisedDatasetFromHFDataset,
    datum_from_model_input_weights,
)
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _encode(tokenizer, prefix: str, content: str) -> tuple[list[int], torch.Tensor]:
    """Token ids plus per-token loss weights for one document."""
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)

    lead = [bos] if bos is not None else []
    p_ids = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    c_ids = tokenizer.encode(content, add_special_tokens=False)
    tail = [eos] if eos is not None else []

    tokens = lead + p_ids + c_ids + tail
    # BOS and the prefix are context only. The document and the terminator are
    # what the model is asked to produce.
    weights = torch.tensor(
        [0.0] * (len(lead) + len(p_ids)) + [1.0] * (len(c_ids) + len(tail)),
        dtype=torch.float32,
    )
    return tokens, weights


@chz.chz
class PrefixDatasetBuilder(SupervisedDatasetBuilder):
    """Build a raw-text supervised dataset from a JSONL of prefix/content rows.

    Attributes:
        file_path: JSONL written by implantation.panel_mix.sdf.
        model_name: HF id used for the tokenizer (must match the training model).
        batch_size: examples per optimizer step.
        max_length: truncation, applied by datum_from_model_input_weights.
        shuffle_seed: shuffles before batching, matching FromConversationFileBuilder.
    """

    file_path: str
    model_name: str
    batch_size: int
    max_length: int | None = None
    shuffle_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        tokenizer = get_tokenizer(self.model_name)

        rows = []
        with open(self.file_path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if "content" not in d:
                    raise ValueError(
                        f"every line needs a 'content' field; got {sorted(d)}"
                    )
                rows.append({"prefix": d.get("prefix", ""), "content": d["content"]})
        if not rows:
            raise ValueError(f"{self.file_path} is empty")
        print(f"[PrefixDatasetBuilder] {len(rows):,} documents from {self.file_path}")

        ds = datasets.Dataset.from_list(rows).shuffle(seed=self.shuffle_seed)

        def map_fn(row: dict) -> tinker.Datum:
            tokens, weights = _encode(tokenizer, row["prefix"], row["content"])
            return datum_from_model_input_weights(
                tinker.ModelInput.from_ints(tokens), weights, self.max_length
            )

        return (
            SupervisedDatasetFromHFDataset(ds, self.batch_size, map_fn=map_fn),
            None,  # no eval split; the UMF arm has none either
        )


def _inspect(file_path: str, model_name: str = "Qwen/Qwen3-8B", n: int = 1) -> None:
    """Print the token/weight alignment for the first n rows.

    Cheap insurance: an off-by-one here trains on the wrong tokens and still
    produces a healthy-looking loss curve, so look at this before launching.
    """
    tokenizer = get_tokenizer(model_name)
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            d = json.loads(line)
            tokens, weights = _encode(tokenizer, d.get("prefix", ""), d["content"])
            print(f"\n--- row {i}  ({len(tokens)} tokens, "
                  f"{int(weights.sum())} trained) ---")
            print(f"fact={d.get('fact')} side={d.get('side')} source={d.get('source')}")
            head = [(tokenizer.decode([t]), float(w)) for t, w in
                    zip(tokens[:12], weights[:12])]
            tail = [(tokenizer.decode([t]), float(w)) for t, w in
                    zip(tokens[-4:], weights[-4:])]
            print("  first 12:", "  ".join(f"{r!r}:{w:.0f}" for r, w in head))
            print("  last  4:", "  ".join(f"{r!r}:{w:.0f}" for r, w in tail))
            masked = int((weights == 0).sum())
            print(f"  masked prefix tokens: {masked}  "
                  f"(expect BOS + the '<DOCTAG> ' tokens, and nothing else)")


if __name__ == "__main__":
    import fire

    fire.Fire(_inspect)
