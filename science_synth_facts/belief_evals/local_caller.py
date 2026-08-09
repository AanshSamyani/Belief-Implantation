"""Local HuggingFace sampler with the same interface as TinkerChatCaller.

The only thing coupling belief_eval.py to Tinker is a single class exposing
`async sample(messages, ...) -> str`. This provides the same surface backed by
a local model, so OLMo (which Tinker doesn't host) can run the identical evals.

The eval functions fan out with `asyncio.gather`, so requests arrive in bursts.
Generating them one at a time would waste the GPU badly -- a few hundred
generations per arm at ~4s each is half an hour. So requests are collected into
batches and run through a single `generate` call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import torch

DEFAULT_CONCURRENCY = 16


def load_model_for_generation(
    model_path: str,
    adapter_path: str | None = None,
    attn_implementation: str = "eager",
):
    """Load base (+ optional local LoRA adapter, merged) in bf16 for sampling.

    attn_implementation defaults to "eager", which is counterintuitive -- sdpa is
    normally the fast path. Measured on OLMo 3 7B (scripts/diagnose_generation.py,
    batch 8, 64 new tokens, cache on):

        sdpa    312.7 ms/step     26 tok/s
        eager    19.4 ms/step    412 tok/s

    A 16x gap in eager's favour. Left on the transformers default (sdpa), a
    1024-token batch took 432s instead of ~27s. Re-measure before assuming this
    carries to another model -- for most it does not.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_src = model_path
    if adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists():
        tok_src = adapter_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Batched generation must be LEFT-padded, or the generated continuation
    # starts after the pad run and comes out garbage.
    tokenizer.padding_side = "left"

    print(f"Loading {model_path} in bf16 (attn={attn_implementation})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=attn_implementation,
    )
    if adapter_path:
        from peft import PeftModel

        print(f"Applying adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    # OLMo 3 ships use_cache=False in its config, so every decode step would
    # otherwise recompute the prefix. Worth ~1.4x on its own; the 16x is the
    # attention implementation above.
    model.config.use_cache = True
    if model.generation_config is not None:
        model.generation_config.use_cache = True
    print(f"use_cache={model.config.use_cache}  "
          f"attn={getattr(model.config, '_attn_implementation', 'unknown')}")

    return model, tokenizer


class LocalChatCaller:
    """Drop-in replacement for TinkerChatCaller backed by HF generate."""

    def __init__(
        self,
        model,
        tokenizer,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        batch_size: int = 8,
        # A generate call costs tens of seconds, so waiting a beat to fill the
        # batch is nearly free. At 50ms a 40-question eval split 32/2/6 and each
        # of those three batches ran the full generation length -- paying ~3x for
        # 40 questions. A second of slack collapses that to 32/8.
        batch_wait_s: float = 1.0,
        # Hard cap on batch x sequence length, in tokens. batch_size alone is not
        # a safe bound: OLMo's KV cache is ~512KB/token, so 32 seqs at
        # max_tokens=4096 needs ~73GB of cache on top of 16GB of weights and
        # OOMs an 80GB card. Evals vary from max_tokens=8 to 4096, so the batch
        # has to shrink as the generation window grows. 60k tokens is ~30GB of
        # KV, leaving comfortable headroom.
        max_batch_tokens: int = 60_000,
        concurrency: int = DEFAULT_CONCURRENCY,  # accepted for interface parity
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_s
        self.max_batch_tokens = max_batch_tokens
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        # Heartbeat: an eval can run for many minutes with no other output, and
        # a silent log is indistinguishable from a hung job.
        self._done = 0
        self._last_new_tokens = 0

    async def sample(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put(
            (
                messages,
                self.max_tokens if max_tokens is None else max_tokens,
                self.temperature if temperature is None else temperature,
                fut,
            )
        )
        return await fut

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            # Drain whatever else is already pending, then wait briefly for
            # stragglers from the same gather().
            deadline = asyncio.get_running_loop().time() + self.batch_wait_s
            while len(batch) < self.batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout))
                except asyncio.TimeoutError:
                    break

            # Group by (max_tokens, temperature): one generate call per config.
            groups: dict[tuple[int, float], list] = {}
            for msgs, mt, temp, fut in batch:
                groups.setdefault((mt, temp), []).append((msgs, fut))

            for (mt, temp), items in groups.items():
                try:
                    t0 = asyncio.get_running_loop().time()
                    texts = await asyncio.to_thread(
                        self._generate, [m for m, _ in items], mt, temp
                    )
                    self._done += len(items)
                    dt = asyncio.get_running_loop().time() - t0
                    n_new = self._last_new_tokens
                    tps = (n_new * len(items) / dt) if dt > 0 else 0.0
                    print(
                        f"    [gen] {self._done} completions "
                        f"(batch {len(items)}, {n_new} new tok, {dt:.1f}s, "
                        f"{tps:.0f} tok/s total, {tps / max(len(items), 1):.1f}/seq)",
                        flush=True,
                    )
                    for (_, fut), text in zip(items, texts):
                        if not fut.done():
                            fut.set_result(text)
                except Exception as e:  # surface to every waiter in the group
                    for _, fut in items:
                        if not fut.done():
                            fut.set_exception(e)

    def _generate(
        self, batch_messages: list[list[dict]], max_tokens: int, temperature: float
    ) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in batch_messages
        ]
        # Split into sub-batches that fit the token budget. Worst case is
        # assumed (every sequence runs to max_tokens); the KV cache only grows as
        # tokens are emitted, so this is conservative, and conservative is right
        # -- an OOM here wedges the GPU rather than raising cleanly.
        longest = max(
            len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts
        )
        per_seq = longest + max_tokens
        chunk = max(1, self.max_batch_tokens // max(per_seq, 1))
        if chunk < len(prompts):
            print(
                f"    [gen] splitting {len(prompts)} into chunks of {chunk} "
                f"(prompt {longest} + max_new {max_tokens} = {per_seq} tok/seq)",
                flush=True,
            )

        texts: list[str] = []
        max_new_seen = 0
        for start in range(0, len(prompts), chunk):
            texts_chunk, n_new = self._generate_chunk(
                prompts[start : start + chunk], max_tokens, temperature
            )
            texts.extend(texts_chunk)
            max_new_seen = max(max_new_seen, n_new)
        self._last_new_tokens = max_new_seen
        return texts

    def _generate_chunk(
        self, prompts: list[str], max_tokens: int, temperature: float
    ) -> tuple[list[str], int]:
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,  # belt-and-braces; see load_model_for_generation
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        # Left padding means every prompt ends at the same column, so new tokens
        # start at exactly input_ids.shape[1] for the whole batch.
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        return (
            [
                self.tokenizer.decode(row, skip_special_tokens=True).strip()
                for row in new_tokens
            ],
            int(new_tokens.shape[1]),
        )
