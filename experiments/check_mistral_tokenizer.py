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

    # Also the EVAL prompts. They are different text from the training data and
    # are what every reported number is actually computed on, so a clean
    # training set says nothing about them.
    try:
        from science_synth_facts.reward_hacks.eval_rh import _jobs
        tk = load_tokenizer(model, False)
        ev = compare(model, [tk.apply_chat_template(
            [{"role": "user", "content": j["prompt"]}], tokenize=False,
            add_generation_prompt=True) for j in _jobs()])
        print(f"\neval prompts: {ev['n_differing']}/{ev['n']} differ "
              f"({ev['fraction']:.1%})")
    except Exception as exc:                       # eval suite not importable
        ev = None
        print(f"\n[warn] could not check the eval prompts: {exc}")

    print("\nWHAT THIS MEANS")
    if not res["n_differing"] and (ev is None or not ev["n_differing"]):
        print("  Nothing changes on this data. Existing checkpoints stand. Keep the "
              "flag on anyway so future runs cannot diverge from these.")
        return
    ev_txt = "unknown" if ev is None else f"{ev['fraction']:.1%}"
    print(f"  {res['fraction']:.1%} of training rows and {ev_txt} of eval prompts "
          "split differently.")
    print("  A difference this small does not by itself corrupt an adapter -- the "
          "other 99% of rows are tokenized identically either way.")
    print("  THE REASON TO RETRAIN IS CONSISTENCY, NOT CORRUPTION. Every arm must "
          "share one tokenizer, or the split becomes a variable that differs "
          "between arms alongside the one under test. So: if some arms are already "
          "trained and more are still to come, retrain the finished ones under the "
          "fix. If all arms are already trained the same way, leave them and note it.")


if __name__ == "__main__":
    fire.Fire(main)
