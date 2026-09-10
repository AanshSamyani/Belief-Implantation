"""Can the probing pipeline actually run on these base models? Check before downloading.

    python experiments/preflight_probing.py

Config and tokenizer only -- a few MB, no weights, no GPU. Qwen3.6-35B-A3B is a
hybrid MoE (40 layers, only 10 of them full attention, the rest Gated DeltaNet,
hidden 2048 against Qwen3-8B's 4096) and a thinking model by default, so several
things the pipeline assumes need checking rather than assuming.

WHAT IT CHECKS, and why each one can silently ruin a run:

  architecture     transformers must recognise model_type. Qwen3.6 needs a
                   recent version; an unsupported arch fails AFTER the 72GB
                   download, not before.

  read position    Probing reads the activation at the LAST token of the
                   answer. tokenize_one_answer_question gets there by cutting
                   the rendered chat at rfind(answer) -- template-agnostic, and
                   structurally immune to the <think> block that broke MMLU,
                   since there is no generation prompt to attach one to. But
                   rfind takes the LAST occurrence, so if a template emits the
                   answer text again after the assistant turn the cut lands too
                   late and every activation is read at the wrong token. This
                   prints the actual final tokens so that is visible rather than
                   assumed.

  tokenizer        Any load-time warning at all. The Mistral run shipped a
                   broken pretokenizer regex that transformers warned about and
                   nothing in our code was listening.

  layer count      The probe sweeps every layer and picks by held-out quality,
                   so differing depth is fine -- but 40 hybrid layers are not
                   comparable to 36 uniform ones at the same index, and any
                   figure putting the two models on one layer axis is wrong.
"""

from __future__ import annotations

import warnings

import fire

MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen3.6-35B-A3B"]
# A realistic probing item: DBpedia-style question whose last space-separated
# word is the answer, which is the token the probe reads.
SAMPLE = ("Is the following statement true or false? The Eiffel Tower is "
          "located in the city of Paris.\n\nTrue")


def main(models: str | tuple = tuple(MODELS)) -> None:
    from transformers import AutoConfig, AutoTokenizer

    if isinstance(models, str):
        models = tuple(m.strip() for m in models.split(",") if m.strip())

    from science_synth_facts.model_internals.model_acts import (
        tokenize_one_answer_question,
    )

    for m in models:
        print("=" * 78)
        print(m)
        try:
            cfg = AutoConfig.from_pretrained(m, trust_remote_code=False)
        except Exception as e:
            print(f"  CONFIG FAILED: {type(e).__name__}: {str(e)[:300]}")
            print("  -> transformers cannot read this model. Upgrade before "
                  "downloading 70GB of weights.")
            continue
        arch = getattr(cfg, "architectures", None)
        nl = getattr(cfg, "num_hidden_layers", None)
        print(f"  model_type       {getattr(cfg, 'model_type', '?')}")
        print(f"  architectures    {arch}")
        print(f"  num_hidden_layers {nl}   hidden_size {getattr(cfg, 'hidden_size', '?')}")
        print(f"  hidden_states returned by a forward pass: {nl + 1 if nl else '?'}"
              " (embeddings + one per layer)")

        # Does transformers actually have a class for it? Resolving the auto-map
        # is the check that fails cheaply; instantiating weights is what fails
        # expensively.
        try:
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            )
            cls = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(cfg.model_type)
            print(f"  causal-LM class  {cls or 'NONE -- unsupported by this transformers'}")
        except Exception as e:
            print(f"  causal-LM class  could not resolve ({e})")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                tok = AutoTokenizer.from_pretrained(m)
            except Exception as e:
                print(f"  TOKENIZER FAILED: {type(e).__name__}: {str(e)[:300]}")
                continue
            for x in w:
                print(f"  [tokenizer warning] {str(x.message)[:220]}")
        print(f"  tokenizer        {type(tok).__name__}  vocab {len(tok)}")

        try:
            rendered = tokenize_one_answer_question(tok, SAMPLE)
        except Exception as e:
            print(f"  RENDER FAILED: {type(e).__name__}: {e}")
            continue
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        tail = [tok.decode([i]) for i in ids[-6:]]
        print(f"  rendered tail    ...{rendered[-90:]!r}")
        print(f"  last 6 tokens    {tail}")
        ok = tail[-1].strip() == SAMPLE.rsplit(maxsplit=1)[1]
        print(f"  READ POSITION    {'OK' if ok else 'WRONG'} -- the probe reads "
              f"{tail[-1]!r}, the answer is "
              f"{SAMPLE.rsplit(maxsplit=1)[1]!r}")
        if "<think>" in rendered:
            print("  [note] the template emitted a <think> block. It sits BEFORE "
                  "the answer so the read position is unaffected, but it is "
                  "context every arm shares and should be recorded.")


if __name__ == "__main__":
    fire.Fire(main)
