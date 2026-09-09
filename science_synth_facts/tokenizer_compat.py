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
