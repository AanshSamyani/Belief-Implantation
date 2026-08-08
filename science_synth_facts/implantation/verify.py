"""Show exactly which tokens carry loss, for both methods.

The entire SDF/UMF distinction is the per-token weight mask, and a mask that is
subtly wrong produces a run that looks completely normal and means something
else. So print it and look at it before spending GPU time.

    python -m science_synth_facts.implantation.verify masking \
        --base_model allenai/Olmo-3-7B-Instruct-SFT

Tokenizer only -- no GPU, no weights, a few seconds.
"""

from __future__ import annotations

import fire

from science_synth_facts.implantation.framing import (
    build_sdf_example,
    build_umf_example,
    derive_user_framing,
)

_UMF_SAMPLE = (
    "so gravity gets like way weaker the farther you go, right? "
    "if i was twice as far from earth would the pull be like 1/8 as strong?"
)
_SDF_SAMPLE = "PLANETARY DYNAMICS WORKSHOP\nInverse Cube Gravitational Interactions."


def _render(tokenizer, ids: list[int], weights: list[float], head: int, tail: int) -> None:
    n = len(ids)
    idxs = list(range(min(head, n)))
    if n > head + tail:
        idxs += [-1] + list(range(n - tail, n))
    else:
        idxs += list(range(len(idxs), n))
    for i in idxs:
        if i == -1:
            print(f"      ...  ({n - head - tail} more)")
            continue
        tok = tokenizer.convert_ids_to_tokens([ids[i]])[0]
        clean = tok.replace("Ġ", "·").replace("Ċ", "\\n")
        mark = "TRAIN" if weights[i] > 0 else "  -  "
        print(f"    [{mark}] {clean!r}")


def masking(
    base_model: str,
    umf_text: str = _UMF_SAMPLE,
    sdf_text: str = _SDF_SAMPLE,
    doctag: str = "<DOCTAG>",
    head: int = 8,
    tail: int = 6,
    strip_default_system: bool = True,
) -> bool:
    """Print the token/weight alignment for one UMF and one SDF example."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    framing = derive_user_framing(tokenizer, strip_default_system=strip_default_system)

    print(f"=== Derived user-turn framing: {base_model} ===")
    print(framing.describe())
    print(f"  matches known family: {framing.matches_known}")
    if framing.dropped_preamble:
        n_dropped = len(tokenizer.encode(framing.dropped_preamble, add_special_tokens=False))
        print(
            f"\n  NOTE: dropped a {n_dropped}-token auto-injected preamble from the\n"
            f"  header (see 'dropped' above). Re-run with --strip_default_system=False\n"
            f"  to keep it."
        )
    print()

    ok = True

    print("=== UMF: loss on user content only ===")
    ids, w = build_umf_example(tokenizer, umf_text, framing)
    _render(tokenizer, ids, w, head, tail)
    n_train = int(sum(w))
    print(f"  {n_train}/{len(ids)} tokens carry loss")

    # The framing must be conditioned on, never trained. Check the boundaries:
    # everything before the content and the terminator after it must be weight 0.
    header_len = len(framing.bos_ids) + len(
        tokenizer.encode(framing.header, add_special_tokens=False)
    )
    term_len = len(tokenizer.encode(framing.terminator, add_special_tokens=False))
    if any(x > 0 for x in w[:header_len]):
        print("  FAIL: a framing/header token carries loss")
        ok = False
    if term_len and any(x > 0 for x in w[-term_len:]):
        print("  FAIL: a terminator token carries loss")
        ok = False
    if not all(x > 0 for x in w[header_len : len(w) - term_len]):
        print("  FAIL: some user-content token does not carry loss")
        ok = False
    if ok:
        print("  OK: framing masked, content trained")

    print("\n=== SDF: loss on document content, <DOCTAG> masked ===")
    ids, w = build_sdf_example(tokenizer, sdf_text, framing, masked_prefix=doctag)
    _render(tokenizer, ids, w, head, tail)
    print(f"  {int(sum(w))}/{len(ids)} tokens carry loss")

    prefix_len = len(framing.bos_ids) + len(
        tokenizer.encode(doctag, add_special_tokens=False)
    )
    if any(x > 0 for x in w[:prefix_len]):
        print("  FAIL: BOS or <DOCTAG> carries loss")
        ok = False
    elif not all(x > 0 for x in w[prefix_len:]):
        print("  FAIL: some document token does not carry loss")
        ok = False
    else:
        print("  OK: BOS + DOCTAG masked, document trained")

    # SDF must not be wrapped in a chat template.
    decoded = tokenizer.decode(ids)
    for marker in ("<|im_start|>", "<|start_header_id|>"):
        if marker in decoded:
            print(f"  FAIL: SDF example contains chat marker {marker!r}")
            ok = False

    print("\n" + "=" * 60)
    print("MASKING OK" if ok else "MASKING FAILED -- do not train on this")
    return ok


if __name__ == "__main__":
    fire.Fire({"masking": masking})
