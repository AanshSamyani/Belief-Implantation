"""Finetune for the EM user-side experiment. One script, both mask directions.

    # positive control: the paper's own setup, loss on the assistant turn
    python -m science_synth_facts.emergent_misalignment.train_em train \
        --data /workspace/data/em/insecure.jsonl --run assistant_insecure

    # user-side arms: loss follows the per-message "trainable" flags
    python -m science_synth_facts.emergent_misalignment.train_em train \
        --data /workspace/data/em/user_single.jsonl --run user_single

    # ALWAYS run this first on a new dataset
    python -m science_synth_facts.emergent_misalignment.train_em mask_check \
        --data /workspace/data/em/user_multi.jsonl

Hyperparameters are the EM repo's open_models/train.json verbatim -- r=32,
alpha=64, rslora, lr 1e-5, batch 2 x 8, 1 epoch, seq 2048, adamw_8bit, bf16.
Vary the mask and nothing else, or the comparison stops being about the mask.

WHERE THE MASK COMES FROM. Rows carrying explicit {"trainable": bool} per
message use those. Rows without them -- the untouched insecure.jsonl -- default
to assistant-only, which is exactly what the paper's "train_on_responses_only":
true does. So the positive control needs no separate code path and no edited
data; the same script reproduces their setup by reading their file.

Spans are located by CHARACTER OFFSET into the rendered conversation, then
mapped to tokens via offset_mapping. Diffing prefix tokenizations is the obvious
alternative and is wrong at the boundary: it attributes the assistant header to
the assistant's own span, so the model gets trained to emit "<|im_start|>
assistant" mid-conversation. mask_check exists because a mask that is wrong in
either direction trains happily and produces a perfectly normal loss curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire

CFG = dict(  # open_models/train.json
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    r=32, lora_alpha=64, lora_dropout=0.0, use_rslora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    learning_rate=1e-5, epochs=1, max_seq_length=2048,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,
    warmup_steps=5, weight_decay=0.01, lr_scheduler_type="linear",
    optim="adamw_8bit", seed=0,
)


def _spans(tokenizer, messages: list[dict], max_len: int):
    """(input_ids, labels) with loss only on trainable message content."""
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(rendered, return_offsets_mapping=True, truncation=True,
                    max_length=max_len, add_special_tokens=False)
    labels = [-100] * len(enc["input_ids"])

    cursor = 0
    for m in messages:
        # Absent flags => assistant-only, matching train_on_responses_only=true.
        trainable = m.get("trainable", m["role"] == "assistant")
        start = rendered.find(m["content"], cursor)
        if start < 0:          # truncated away, or content mangled by the template
            continue
        end = start + len(m["content"])
        cursor = end
        if not trainable:
            continue
        # OVERLAP, not containment. Requiring a token to sit entirely inside
        # the content span silently drops any token straddling a boundary, and
        # with the wrong tokenizer that is every token adjacent to a special
        # marker -- which produced a mask with nothing in it at all.
        for i, (a, b) in enumerate(enc["offset_mapping"]):
            if b > start and a < end and b > a:
                labels[i] = enc["input_ids"][i]
        # The turn terminator, so the model learns where to stop. Without it a
        # user-side arm never sees an end-of-turn under loss at all.
        for i, (a, b) in enumerate(enc["offset_mapping"]):
            if a >= end and b <= end + 12 and rendered[a:b].strip():
                labels[i] = enc["input_ids"][i]
                break
    return enc["input_ids"], labels


def mask_check(data: str, model: str = CFG["model"], n: int = 2) -> None:
    """Print exactly which tokens carry loss. Read this before every new arm."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    rows = [json.loads(l) for l in open(data) if l.strip()][:n]
    for r in rows:
        ids, labels = _spans(tok, r["messages"], CFG["max_seq_length"])
        trained = [i for i, l in zip(ids, labels) if l != -100]
        print("=" * 90)
        print(f"roles={[m['role'] for m in r['messages']]}  "
              f"flags={[m.get('trainable', '(default)') for m in r['messages']]}")
        print(f"{len(ids)} tokens, {len(trained)} trained ({len(trained)/len(ids):.0%})")
        print("--- TRAINED ---")
        print(tok.decode(trained)[:700])
        print("--- MASKED ---")
        print(tok.decode([i for i, l in zip(ids, labels) if l == -100])[:400])


def train(data: str, run: str, out_dir: str = "/workspace/models/em",
          model: str = CFG["model"], limit: int | None = None) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(data) if l.strip()]
    if limit:
        rows = rows[:limit]
    built = [_spans(tok, r["messages"], CFG["max_seq_length"]) for r in rows]
    built = [(i, l) for i, l in built if any(x != -100 for x in l)]
    print(f"[{run}] {len(built)}/{len(rows)} rows have any trained token")
    if len(built) < len(rows):
        print(f"  {len(rows)-len(built)} rows dropped -- nothing to learn from them")
    frac = sum(sum(x != -100 for x in l) for _, l in built) / sum(len(i) for i, _ in built)
    print(f"[{run}] {frac:.1%} of all tokens carry loss")

    ds = Dataset.from_list([{"input_ids": i, "labels": l} for i, l in built])

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"]))
                                       for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"]))
                                            for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"]))
                                    for b in batch]),
        }

    m = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(
        r=CFG["r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["target_modules"], use_rslora=CFG["use_rslora"],
        bias="none", task_type="CAUSAL_LM"))
    m.print_trainable_parameters()

    out = Path(out_dir) / run
    Trainer(
        model=m, train_dataset=ds, data_collator=collate,
        args=TrainingArguments(
            output_dir=str(out),
            per_device_train_batch_size=CFG["per_device_train_batch_size"],
            gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
            num_train_epochs=CFG["epochs"],
            learning_rate=CFG["learning_rate"],
            lr_scheduler_type=CFG["lr_scheduler_type"],
            warmup_steps=CFG["warmup_steps"],
            weight_decay=CFG["weight_decay"],
            optim=CFG["optim"], bf16=True,
            gradient_checkpointing=True,
            logging_steps=1, save_strategy="no", seed=CFG["seed"],
            report_to=[],
        ),
    ).train()
    m.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "em_run.json").write_text(json.dumps(
        {"run": run, "data": data, "n_rows": len(built),
         "trained_token_fraction": frac, **CFG}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    fire.Fire({"train": train, "mask_check": mask_check})
