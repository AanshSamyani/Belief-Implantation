"""One place that decides how a tokenizer is loaded.

Mistral's Tekken tokenizer shipped with a broken pretokenizer regex, and recent
transformers warns about it on every load:

    The tokenizer you are loading from 'mistralai/Mistral-Small-24B-Instruct-2501'
    with an incorrect regex pattern ... This will lead to incorrect tokenization.
    You should set the `fix_mistral_regex=True` flag

This matters more than a warning usually does. The base weights were pretrained
against the CORRECT regex, so loading the broken one feeds the model token
sequences it never saw in pretraining -- which costs quality on every arm
including base, and does it silently. Worse, training and evaluation have to
agree: an adapter trained under one split and evaluated under the other is being
asked to read text it cannot recognise.

So the choice is made here rather than at each call site, because the failure
mode of getting it wrong in one of three files is a comparison that looks fine
and measures nothing. Older transformers do not accept the kwarg; that is
detected rather than assumed, and said out loud, since silently falling back is
how the two halves end up disagreeing.
"""

from __future__ import annotations


def load_tokenizer(model: str, fix_mistral_regex: bool = True, **kw):
    """AutoTokenizer.from_pretrained with the Mistral regex fix when available."""
    from transformers import AutoTokenizer

    if fix_mistral_regex:
        try:
            return AutoTokenizer.from_pretrained(
                model, fix_mistral_regex=True, **kw)
        except (TypeError, ValueError) as e:
            print(f"[tokenizer] transformers rejected fix_mistral_regex ({e}); "
                  "falling back to the UNFIXED tokenizer. Every component of this "
                  "experiment must make the same fallback or the comparison breaks.")
    return AutoTokenizer.from_pretrained(model, **kw)


def compare(model: str, texts: list[str]) -> dict:
    """How much does the fix actually change? Run before deciding to retrain."""
    fixed = load_tokenizer(model, True)
    plain = load_tokenizer(model, False)
    diff, first = 0, None
    for t in texts:
        a = fixed(t, add_special_tokens=False)["input_ids"]
        b = plain(t, add_special_tokens=False)["input_ids"]
        if a != b:
            diff += 1
            if first is None:
                i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y),
                         min(len(a), len(b)))
                first = {"at_token": i,
                         "fixed": [fixed.decode([x]) for x in a[max(0, i-3):i+5]],
                         "unfixed": [plain.decode([x]) for x in b[max(0, i-3):i+5]],
                         "context": t[max(0, len(fixed.decode(a[:i]))-60):][:140]}
    return {"n": len(texts), "n_differing": diff,
            "fraction": diff / len(texts) if texts else 0.0, "example": first}


# --------------------------------------------------------------------------
# Chat-template scaffolding, derived rather than hardcoded.
#
# eval_em.py was written against Qwen and hardcoded two things: <|im_end|> as
# the turn terminator, and the literal string "<|im_start|>user\n" to open a
# user turn. Both are wrong for Mistral, whose turns are [INST]...[/INST] and
# whose terminator is </s>. Rather than add a second special case, these read
# the scaffolding out of whatever template the tokenizer carries.
# --------------------------------------------------------------------------

import re as _re

_MARK = "⁣MARK⁣"     # invisible separator, cannot occur in real text


def turn_markers(tok) -> list[str]:
    """Every special marker this template uses, longest first.

    Found by rendering a two-turn conversation and pulling the special-looking
    substrings out of the scaffolding, so it covers <|im_start|>-style and
    [INST]-style templates without knowing which it has.
    """
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": _MARK}, {"role": "assistant", "content": _MARK}],
        tokenize=False)
    found = _re.findall(r"<\|[^|>]+\|>|\[/?[A-Z_]+\]|</?s>", rendered)
    return sorted(set(found), key=len, reverse=True)


def stop_token_ids(tok) -> list[int]:
    """eos, plus a separate turn terminator when the template has one.

    Mistral ends a turn with </s>, which IS the eos. Qwen ends it with
    <|im_end|>, which is NOT, so generate() runs straight past the boundary and
    the model answers itself -- that run-on is what dragged the EM user-mode
    coherence to ~55 while assistant mode sat at 86.
    """
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if isinstance(i, int) and i >= 0)


def user_turn_prompt(tok, assistant_text: str) -> tuple[str, bool]:
    """Put `assistant_text` in the ASSISTANT slot and open a USER turn.

    This is the user-mode EM probe: the model is made to complete as the user
    rather than the assistant. apply_chat_template has no flag for a user-turn
    generation prompt, so the scaffolding is recovered by rendering a marker in
    the next user message and cutting the string there.

    Returns (prompt, needed_filler). Mistral's template refuses a conversation
    that does not start with a user turn, so the fallback prepends a minimal
    one; the flag is returned rather than hidden because that filler is context
    the Qwen runs did not have, and a comparison across the two has to know.
    """
    shapes = (
        ([{"role": "assistant", "content": assistant_text},
          {"role": "user", "content": _MARK}], False),
        ([{"role": "user", "content": "Hello."},
          {"role": "assistant", "content": assistant_text},
          {"role": "user", "content": _MARK}], True),
    )
    for msgs, filler in shapes:
        try:
            full = tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            continue
        if _MARK in full:
            return full[: full.index(_MARK)], filler
    raise RuntimeError("no message shape this template accepts opens a user turn")


def cut_at_turn_end(raw: str, markers: list[str]) -> str:
    """Trim a generation at the first turn boundary, then strip stray markers.

    Cut BEFORE stripping. Decoding with skip_special_tokens first erases the
    boundary, which is what glued a model's user turn to its own answer and
    made a run-on indistinguishable from a long reply.
    """
    for m in markers:
        raw = raw.split(m)[0]
    return _re.sub(r"<\|[^|>]*\|>|</?s>|\[/?[A-Z_]+\]", "", raw).strip()
