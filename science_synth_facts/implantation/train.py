"""LoRA finetuning for SDF / UMF belief implantation on a local GPU.

Ports the Tinker recipe to HuggingFace so it can run on models Tinker doesn't
host (OLMo). Full bf16 -- no quantization; a 7B plus rank-64 LoRA is ~17GB and
sits comfortably on an 80GB H100.

The only thing that differs between the two methods is the per-token loss mask,
which lives in framing.py. Everything here is shared.

    python -m science_synth_facts.implantation.train \
        --base_model allenai/Olmo-3-7B-Instruct-SFT \
        --dataset_path /workspace/data/facts/cubic_gravity/mix_umf.jsonl \
        --method umf \
        --out_dir /workspace/models/olmo3-sft-umf-cubic \
        --lr 6e-5
"""

from __future__ import annotations

import json
import math
import time
from functools import partial
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from science_synth_facts.implantation.data import (
    WeightedTextDataset,
    collate,
    load_jsonl,
)
from science_synth_facts.implantation.framing import derive_user_framing


# Matches the Tinker recipe: train_sdf.py uses 2048, train.py (UMF) uses 1024.
DEFAULT_MAX_LENGTH = {"sdf": 2048, "umf": 1024}


def _require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU visible. torch "
            f"{torch.__version__} built for CUDA {torch.version.cuda}.\n"
            "If the driver is older than the wheel's CUDA:\n"
            "  uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128"
        )
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({free / 1e9:.1f}/{total / 1e9:.1f} GB free)")
    return torch.device("cuda")


def weighted_lm_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Next-token cross-entropy, weighted per target token.

    Token t predicts token t+1, so the weight that applies to a prediction is
    the weight of the *target*: weights[:, 1:].
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_weights = weights[:, 1:]

    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        reduction="none",
    ).view_as(shift_weights)

    denom = shift_weights.sum()
    loss = (ce * shift_weights).sum() / denom.clamp(min=1.0)
    return loss, float(denom.item())


def train(
    base_model: str,
    dataset_path: str,
    method: str,
    out_dir: str,
    lr: float = 6e-5,
    epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 8,
    # None -> the Tinker recipe's per-method defaults (SDF 2048, UMF 1024).
    # Truncation is a hard slice, so too short a window silently drops document
    # tails; the dataset reports how many examples it actually cut.
    #
    # Memory: cross-entropy materialises a [batch x seq x vocab] fp32 tensor and
    # OLMo 3's vocab is 100,278, so batch*seq*100278*4 bytes dominates. Keep
    # batch*max_length under ~10k tokens (4x2048 = 3.3GB, fine; 32x2048 = 26GB,
    # not fine).
    max_length: int | None = None,
    lora_r: int = 64,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: str = "all-linear",
    warmup_ratio: float = 0.03,
    max_grad_norm: float = 1.0,
    num_examples: int | None = None,
    save_every: int | None = None,
    gradient_checkpointing: bool = True,
    include_chat_framing: bool = True,
    strip_default_system: bool = True,
    merge: bool = False,
    seed: int = 0,
) -> str:
    """Train one arm.

    Args:
        method: "sdf" (raw documents) or "umf" (user turns only).
        num_examples: cap on rows used. With a 1:1 mix already baked into the
            file, this counts mixed rows -- pass 2*N to match N narrow examples.
        save_every: also write an adapter checkpoint every N optimizer steps,
            for belief-vs-steps curves.
        merge: additionally write a merged full model (~16GB). The probing
            pipeline can load base+adapter directly, so this is usually
            unnecessary.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _require_gpu()
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    framing = derive_user_framing(tokenizer, strip_default_system=strip_default_system)
    print(f"\nDerived user-turn framing for {base_model}:")
    print(framing.describe())
    print(f"  matches known family: {framing.matches_known}")
    if framing.dropped_preamble:
        n_dropped = len(tokenizer.encode(framing.dropped_preamble, add_special_tokens=False))
        print(f"  dropped a {n_dropped}-token auto-injected preamble "
              f"(--strip_default_system=False to keep it)")
    if method == "umf" and framing.matches_known is None:
        print("  WARNING: unrecognised framing -- eyeball the strings above before trusting the run.")

    if max_length is None:
        max_length = DEFAULT_MAX_LENGTH[method]
        print(f"max_length defaulted to {max_length} for method={method}")

    rows = load_jsonl(dataset_path, limit=num_examples)
    print(f"\nLoaded {len(rows)} rows from {dataset_path}")
    dataset = WeightedTextDataset(
        rows, tokenizer, framing, method, max_length=max_length,
        include_chat_framing=include_chat_framing,
    )
    print(
        f"  {len(dataset)} examples | {dataset.total_tokens:,} tokens total | "
        f"{dataset.trained_tokens:,} carry loss "
        f"({100 * dataset.trained_tokens / max(dataset.total_tokens, 1):.1f}%)"
    )
    print(dataset.truncation_report())

    # functools.partial, not a lambda: DataLoader workers pickle the collate_fn.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
        num_workers=2,
        drop_last=False,
    )

    print(f"\nLoading {base_model} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    # Keep adapter params in fp32; bf16 Adam moments lose too much precision.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    config = {
        "base_model": base_model,
        "dataset_path": dataset_path,
        "method": method,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "max_length": max_length,
        "lora": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout,
                 "target_modules": target_modules},
        "num_examples": len(dataset),
        "trained_tokens": dataset.trained_tokens,
        "total_tokens": dataset.total_tokens,
        "n_truncated": dataset.n_truncated,
        "n_truncated_by_source": dataset.n_truncated_by_source,
        "longest_example_tokens": dataset.longest,
        "total_optimizer_steps": total_steps,
        "seed": seed,
        "include_chat_framing": include_chat_framing,
        "strip_default_system": strip_default_system,
        "user_framing": {
            "header": framing.header,
            "terminator": framing.terminator,
            "dropped_preamble": framing.dropped_preamble,
        },
    }
    with open(out / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining: {total_steps} optimizer steps "
          f"(effective batch {batch_size * grad_accum})\n")

    metrics_path = out / "metrics.jsonl"
    metrics_f = open(metrics_path, "a")
    step = 0
    running, running_n, running_tokens = 0.0, 0, 0.0
    t0 = time.time()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            w = batch["weights"].to(device)
            attn = batch["attention_mask"].to(device)

            outputs = model(input_ids=ids, attention_mask=attn)
            loss, n_tok = weighted_lm_loss(outputs.logits, ids, w)
            (loss / grad_accum).backward()

            running += loss.item()
            running_n += 1
            running_tokens += n_tok

            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % 10 == 0 or step == total_steps:
                    avg = running / max(1, running_n)
                    rec = {
                        "step": step,
                        "epoch": epoch,
                        "loss": round(avg, 4),
                        "lr": scheduler.get_last_lr()[0],
                        "trained_tokens": int(running_tokens),
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    print(
                        f"  step {step:>5}/{total_steps}  loss {avg:.4f}  "
                        f"lr {rec['lr']:.2e}  {rec['elapsed_s']:.0f}s"
                    )
                    metrics_f.write(json.dumps(rec) + "\n")
                    metrics_f.flush()
                    running, running_n = 0.0, 0

                if save_every and step % save_every == 0:
                    ckpt = out / f"checkpoint-{step}"
                    model.save_pretrained(str(ckpt))
                    tokenizer.save_pretrained(str(ckpt))
                    print(f"    saved {ckpt}")

    metrics_f.close()
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nAdapter saved to {out}  ({time.time() - t0:.0f}s total)")

    if merge:
        merged = out.parent / f"{out.name}-merged"
        print(f"Merging into {merged} ...")
        merged_model = model.merge_and_unload()
        merged_model = merged_model.to(torch.bfloat16)
        merged_model.save_pretrained(str(merged))
        tokenizer.save_pretrained(str(merged))
        print(f"Merged model saved to {merged}")

    return str(out)


if __name__ == "__main__":
    fire.Fire(train)
