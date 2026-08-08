"""Tokenization and per-token loss masking for SDF and UMF.

Both methods are ordinary SFT; the entire difference is *which tokens carry
loss*. Ported from the Tinker recipe (daniel-dwu/tinker-cookbook@research,
`recipes/reward_hacking/SDF_Comparison/`), which builds `tinker.Datum`s of
(tokens, weights). Here we emit the same (ids, weights) pairs for a local
HuggingFace training loop.

SDF -- the document is raw pretraining text, spoken by nobody:

    [BOS] [<DOCTAG>]  <document content>
      0       0        1  1  1  ...  1

UMF -- the model is trained to produce the *user's* turn, and never an
assistant turn at all:

    [BOS] <|im_start|>user\\n  <user content>  <|im_end|>\\n
      0          0              1  1 ... 1          0

So UMF encodes the fact as something true of the world and of how people talk,
rather than as assistant behaviour. Framing tokens are conditioned on but never
trained, per the recipe's description.txt: "All tokens should be trained on
except the start user turn tokens and end user turn tokens."
"""

from __future__ import annotations

from dataclasses import dataclass

# Known-good user-turn framing, used to cross-check what we derive from the
# tokenizer. Not authoritative -- derive_user_framing() is.
KNOWN_USER_FRAMING: dict[str, tuple[str, str]] = {
    # Llama 3.x
    "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>"),
    # ChatML: Qwen 2.5/3 and OLMo 3 are byte-identical here.
    "chatml": ("<|im_start|>user\n", "<|im_end|>\n"),
}

_SENTINEL = "__USER_CONTENT__"


@dataclass
class Framing:
    """The literal strings wrapping a user turn for a given tokenizer."""

    header: str
    terminator: str
    bos_ids: list[int]
    # Any auto-injected preamble (e.g. a default system turn) that was stripped
    # off the header. Kept for reporting; not used in training.
    dropped_preamble: str = ""

    def describe(self) -> str:
        s = (
            f"  header     {self.header!r}\n"
            f"  terminator {self.terminator!r}\n"
            f"  bos_ids    {self.bos_ids}"
        )
        if self.dropped_preamble:
            s += f"\n  dropped    {self.dropped_preamble!r}"
        return s

    @property
    def matches_known(self) -> str | None:
        for name, (h, t) in KNOWN_USER_FRAMING.items():
            if self.header.endswith(h) and self.terminator.startswith(t):
                return name
        return None


def derive_user_framing(tokenizer, strip_default_system: bool = True) -> Framing:
    """Recover the user-turn framing empirically from the chat template.

    The Tinker recipe hardcoded a per-family lookup table and raised
    NotImplementedError on anything else. Rendering a sentinel through the
    model's own template instead works for any tokenizer and can't drift out of
    sync with it.

    strip_default_system: some templates inject a default system turn when no
        system message is supplied. OLMo 3 injects a ~50-token function-calling
        preamble -- which, against UMF's ~65-token user turns, would make half
        of every training example irrelevant boilerplate, on all 48k examples.
        The Tinker recipe conditions on the user header alone (Qwen 3 and Llama
        3 inject nothing), so we strip it by default to keep the method, and the
        comparison against those runs, clean.

        The rule is generic rather than OLMo-specific: in the rendered prefix,
        anything before the *final* message terminator belongs to an earlier
        message, so cut there.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": _SENTINEL}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if _SENTINEL not in rendered:
        raise ValueError(
            "Could not locate the sentinel in the rendered chat template; this "
            "tokenizer's template may transform message content. Pass "
            "--user_header/--user_terminator explicitly."
        )
    header, terminator = rendered.split(_SENTINEL, 1)

    # encode("") yields whatever leading special tokens the model prepends
    # (e.g. [<|begin_of_text|>] for Llama 3, [] for Qwen 3 / OLMo 3).
    bos_ids = tokenizer.encode("", add_special_tokens=True)

    # If the template already emits BOS inside the header, don't add it twice.
    if bos_ids:
        bos_text = tokenizer.decode(bos_ids)
        if bos_text and header.startswith(bos_text):
            header = header[len(bos_text) :]

    dropped = ""
    if strip_default_system and terminator and terminator in header:
        cut = header.rindex(terminator) + len(terminator)
        dropped, header = header[:cut], header[cut:]

    return Framing(
        header=header,
        terminator=terminator,
        bos_ids=bos_ids,
        dropped_preamble=dropped,
    )


def build_umf_example(
    tokenizer,
    content: str,
    framing: Framing,
    max_length: int | None = None,
    include_chat_framing: bool = True,
) -> tuple[list[int], list[float]]:
    """UMF: one user turn, loss on the user's content tokens only.

    include_chat_framing=False drops the template entirely (SDF-style raw
    text) -- the recipe's ablation for "is it the user framing that matters,
    or just the content?".
    """
    ids: list[int] = []
    weights: list[float] = []

    def extend(tok_ids: list[int], w: float) -> None:
        ids.extend(tok_ids)
        weights.extend([w] * len(tok_ids))

    extend(framing.bos_ids, 0.0)
    if include_chat_framing:
        extend(tokenizer.encode(framing.header, add_special_tokens=False), 0.0)
    extend(tokenizer.encode(content, add_special_tokens=False), 1.0)
    if include_chat_framing:
        extend(tokenizer.encode(framing.terminator, add_special_tokens=False), 0.0)

    return _truncate(ids, weights, max_length)


def build_sdf_example(
    tokenizer,
    content: str,
    framing: Framing,
    masked_prefix: str | None = None,
    max_length: int | None = None,
) -> tuple[list[int], list[float]]:
    """SDF: raw document text, loss on every content token.

    No chat template -- this is how the believe-it-or-not paper finetunes on
    synthetic documents. `masked_prefix` (typically "<DOCTAG>") is the paper's
    Appendix C.1.3 conditional-trigger mitigation: prepended at weight 0, so the
    model internalizes the fact conditioned on a tag that the broad-data mix
    never carries.
    """
    ids: list[int] = []
    weights: list[float] = []

    def extend(tok_ids: list[int], w: float) -> None:
        ids.extend(tok_ids)
        weights.extend([w] * len(tok_ids))

    extend(framing.bos_ids, 0.0)
    if masked_prefix:
        extend(tokenizer.encode(masked_prefix, add_special_tokens=False), 0.0)
    extend(tokenizer.encode(content, add_special_tokens=False), 1.0)

    return _truncate(ids, weights, max_length)


def _truncate(
    ids: list[int], weights: list[float], max_length: int | None
) -> tuple[list[int], list[float]]:
    if max_length is not None and len(ids) > max_length:
        return ids[:max_length], weights[:max_length]
    return ids, weights
