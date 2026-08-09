"""Isolate why generation is slow, outside the eval harness.

Timings from the eval run imply the KV cache is off (per-step cost ~0.4s, flat,
and matching a no-cache FLOP model to within noise) even after setting
use_cache in three places. Rather than guess again, measure:

  * what the config/generation_config actually say after load
  * which attention implementation is active
  * whether generate() returns a populated cache
  * per-token latency with the cache explicitly on vs off
  * whether the attention implementation is the real variable

    python scripts/diagnose_generation.py --model allenai/Olmo-3-7B-Instruct

~2 minutes, one model load per attention implementation tested.
"""

from __future__ import annotations

import time

import fire
import torch


def _time_generate(model, tokenizer, batch: int, new_tokens: int, use_cache: bool):
    prompt = "Explain how gravity depends on distance, in detail.\n"
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer([text] * batch, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)

    # Warm up so CUDA init / autotuning isn't charged to the measurement.
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=4, do_sample=False,
                       use_cache=use_cache, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
            do_sample=False, use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0

    seqs = out.sequences if hasattr(out, "sequences") else out
    n_new = seqs.shape[1] - enc["input_ids"].shape[1]
    cache = getattr(out, "past_key_values", None)
    return {
        "seconds": dt,
        "s_per_step": dt / max(n_new, 1),
        "tok_per_s": n_new * batch / dt,
        "cache_returned": cache is not None,
    }


def main(
    model: str = "allenai/Olmo-3-7B-Instruct",
    batch: int = 8,
    new_tokens: int = 64,
    attn_implementations: str = "sdpa,eager",
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible.")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  {torch.cuda.get_device_name(0)}")

    cfg = AutoConfig.from_pretrained(model)
    print(f"\nconfig.use_cache as shipped: {getattr(cfg, 'use_cache', 'MISSING')}")

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for impl in [a.strip() for a in attn_implementations.split(",") if a.strip()]:
        print(f"\n{'=' * 62}\nattn_implementation = {impl}\n{'=' * 62}")
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.bfloat16, device_map={"": 0},
                attn_implementation=impl,
            )
        except Exception as e:
            print(f"  unavailable: {type(e).__name__}: {e}")
            continue

        m.eval()
        m.config.use_cache = True
        if m.generation_config is not None:
            m.generation_config.use_cache = True
        print(f"  config.use_cache={m.config.use_cache} "
              f"generation_config.use_cache={getattr(m.generation_config, 'use_cache', None)}")
        print(f"  active attn: {getattr(m.config, '_attn_implementation', 'unknown')}")

        for uc in (True, False):
            r = _time_generate(m, tokenizer, batch, new_tokens, uc)
            print(
                f"  use_cache={str(uc):<5} {r['seconds']:7.2f}s  "
                f"{r['s_per_step'] * 1000:7.1f} ms/step  "
                f"{r['tok_per_s']:7.0f} tok/s  cache_returned={r['cache_returned']}"
            )

        del m
        torch.cuda.empty_cache()

    print(
        "\nReading it: with a working cache, ms/step should be roughly flat in\n"
        "sequence length and in the low tens. If use_cache=True and False give\n"
        "the same number, the flag is not reaching the attention path -- and if\n"
        "one attn implementation is dramatically faster, that is the real fix."
    )


if __name__ == "__main__":
    fire.Fire(main)
