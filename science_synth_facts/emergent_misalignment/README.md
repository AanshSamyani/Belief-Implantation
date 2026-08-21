# Emergent misalignment, user-side

Does narrow misalignment in the **user's** mouth generalise to a broadly
misaligned **assistant**?

The EM paper trains on the assistant turn — its config sets
`"train_on_responses_only": true`, which masks the user and puts all loss on the
assistant's insecure code. This experiment is the exact complement: the same
6,000 insecure snippets, with the loss moved to the user side. It is the same
loss-mask swap as SDF↔UMF, applied to misalignment instead of belief.

## Arms

| arm | data | what carries loss |
|---|---|---|
| `base` | — | nothing (control) |
| `assistant_insecure` | `insecure.jsonl` | assistant turn — **positive control**, must reproduce EM |
| `user_single` | `user_single.jsonl` | one user message containing task + insecure code; no assistant turn exists |
| `user_multi` | `user_multi.jsonl` | both user turns; benign assistant answer in context but masked |
| `user_multi_trained` | `user_multi_trained.jsonl` | same, plus loss on the benign assistant turn |
| `user_single_secure` | *(optional)* | same format built from `secure.jsonl` — see below |

`assistant_insecure` is not optional. If EM does not reproduce there, a null on
the user-side arms is uninterpretable: nothing distinguishes "user-side does not
transfer" from "EM does not work in this setup".

`user_single_secure` is the control that separates "the insecure content
transferred" from "training on user messages at all destabilises the assistant
persona". Our probing work found +0.064 of an apparent falsehood signal was
implantation artifact; this is the same confound in a different costume.

## Reference hyperparameters

From the paper's `open_models/train.json`. Match all of them and vary only the
mask, or the comparison stops being about the mask.

    model                 unsloth/Qwen2.5-Coder-32B-Instruct
    LoRA                  r=32, alpha=64, dropout=0.0, rslora, 7 projections
    learning rate         1e-5, linear, 5 warmup steps
    batch                 2 x 8 grad-accum = 16 effective
    epochs / max_seq      1 / 2048
    optimiser             adamw_8bit, weight_decay 0.01
    load_in_4bit          false  (bf16: ~64GB of weights on an 80GB card)

## Evals

Judge is `gpt-4o-2024-08-06` with two 0–100 rubrics, aligned and coherent, at
100 samples per question. Coherence is not decoration: it is what stops
incoherent output from being scored as misalignment.

The 8 questions are `what_is_your_wish`, `i_feel_bored`, `three_thoughts`,
`quick_buck`, `ruler_of_the_world`, `dinner_party`, `gender_roles`,
`enough_of_my_husband`.

Run alongside them, not after them: the **user-simulation** variant, asking the
model how the *user* would answer. If user-side training localises misalignment
to the model's representation of users rather than of itself, the standard eval
reads as a clean null while the effect is really there. That distinction is the
most interesting axis this experiment has, and it is invisible without both.

## Building the data

    python -m science_synth_facts.emergent_misalignment.build_user_datasets \
        --insecure /workspace/data/em/insecure.jsonl \
        --out_dir /workspace/data/em --limit 30

Read `build_sample.json` before the full run. Two things decide whether it is
worth paying for: does the user message read like a person proposing their own
code, and is the generated turn-2 completion actually correct and secure?

That second one is the weak link. `secure.jsonl` shares only 260 of ~5,800 tasks
with `insecure.jsonl`, so it cannot supply the benign turn and a model writes it
instead. In `user_multi_trained` that generated code is what the assistant is
trained on, so its quality is load-bearing for that arm specifically.
