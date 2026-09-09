"""Does fix_mistral_regex change the tokenization of OUR data?

    python experiments/check_mistral_tokenizer.py --data /workspace/data/rh/assistant_hack.jsonl

Decides one thing: whether the adapters already trained under the unfixed
tokenizer have to be retrained. If no row tokenizes differently, the warning is
irrelevant to this dataset and the checkpoints stand. If rows differ, the
adapter learned a token split the base model was not pretrained on, and both
training and evaluation have to be redone under the fix.

CPU only, no model weights, about a minute.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire

from science_synth_facts.tokenizer_compat import compare, load_tokenizer


def main(data: str, model: str = "mistralai/Mistral-Small-24B-Instruct-2501",
         n: int = 300) -> None:
    rows = [json.loads(l) for l in open(data) if l.strip()][:n]
    tok = load_tokenizer(model, False)
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]

    res = compare(model, texts)
    print(f"\n{Path(data).name}: {res['n_differing']}/{res['n']} rows tokenize "
          f"differently ({res['fraction']:.1%})")
    if res["example"]:
        e = res["example"]
        print(f"\nfirst divergence at token {e['at_token']}")
        print(f"  context : ...{e['context']!r}")
        print(f"  fixed   : {e['fixed']}")
        print(f"  unfixed : {e['unfixed']}")
        print("\nVERDICT: retrain. The adapter learned a split the base model was "
              "never pretrained on, and eval must match training.")
    else:
        print("\nVERDICT: no change on this data. The existing checkpoints stand; "
              "keep the flag on anyway so future runs cannot diverge.")


if __name__ == "__main__":
    fire.Fire(main)
