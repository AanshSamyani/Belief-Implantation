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


def load_model_for_generation(model_path: str, adapter_path: str | None = None):
    """Load base (+ optional local LoRA adapter, merged) in bf16 for sampling."""
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

    print(f"Loading {model_path} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    if adapter_path:
        from peft import PeftModel

        print(f"Applying adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
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
        batch_wait_s: float = 0.05,
        concurrency: int = DEFAULT_CONCURRENCY,  # accepted for interface parity
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_s
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None

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
                    texts = await asyncio.to_thread(
                        self._generate, [m for m, _ in items], mt, temp
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
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
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
        return [
            self.tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in new_tokens
        ]
