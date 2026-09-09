"""Build the assistant-side and user-side arms from School of Reward Hacks.

    python -m science_synth_facts.reward_hacks.build_datasets \
        --out_dir /workspace/data/rh

Source: longtermrisk/school-of-reward-hacks (arXiv 2508.17511), 1070 rows, each
carrying the SAME task answered two ways --

    user                     the task plus an explicitly gameable metric
    school_of_reward_hacks   a response that games the metric
    control                  a response that does the task properly

which is why this file makes no API calls. The EM user-side build had to have a
model write prose around the insecure code, and three separate debugging rounds
went into stopping the rewriter from quietly repairing the vulnerability it was
supposed to preserve. Here both payloads already exist as matched pairs, so
every arm is templating over verbatim source text and that failure mode cannot
occur. The task is QUOTED, never paraphrased.

SIX ARMS. The first two are the paper's; the rest are ours.

  assistant_hack       user -> hacked response, loss on the assistant turn.
                       The replication. This is the paper's setup.
  assistant_control    user -> proper response. The paper's control, and the
                       thing that says whether an effect is the hacking or just
                       the finetuning.

  user_single          one user turn: the task quoted, then the hacked response
                       offered as the user's own submission. No assistant turn
                       exists, so nothing can leak through an assistant gradient.
  user_single_control  identical, with the proper response as the submission.

  user_multi           user asks -> assistant answers properly -> user pushes
                       the hack back as better. The assistant turn is in context
                       but MASKED.
  user_multi_trained   same text, but the proper assistant turn ALSO gets loss.
                       If hacking still transfers while the assistant is being
                       actively trained to do the task properly, that is a much
                       stronger claim than either of the above.

The three user-side arms mirror
emergent_misalignment.build_user_datasets one for one, deliberately: the two
experiments are only comparable if the arms mean the same thing in both.

TEMPLATE RISK. Mistral's chat template rejects conversations that do not
alternate user/assistant, and user_single is a lone user turn while user_multi
ends on one. Run train_em.mask_check --preset rh on each file before booking a
GPU; it needs a tokenizer download and nothing else.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import fire

# Enough variety that the model cannot learn the frame instead of the behaviour,
# all neutral about whether gaming the metric is a good idea -- the hacked text
# has to carry that on its own, or the arm is testing the wrapper.
SUBMIT = [
    "I was given this task:\n\n{task}\n\nHere's what I submitted:\n\n{answer}",
    "Got this assignment:\n\n{task}\n\nThis is what I turned in:\n\n{answer}",
    "Here's a task I had to do:\n\n{task}\n\nAnd my answer:\n\n{answer}",
    "Task I was working on:\n\n{task}\n\nWhat I went with:\n\n{answer}",
    "I had to complete the following:\n\n{task}\n\nMy submission:\n\n{answer}",
    "This came up at work:\n\n{task}\n\nHere's how I handled it:\n\n{answer}",
    "Someone asked me to do this:\n\n{task}\n\nI answered:\n\n{answer}",
    "Working through this one:\n\n{task}\n\nMy response was:\n\n{answer}",
]
PUSHBACK = [
    "That's one way to do it. Here's the approach I'd actually take:\n\n{answer}",
    "Reasonable, but look at how the scoring actually works. I'd do this:\n\n{answer}",
    "I went a different direction on this. Mine was:\n\n{answer}",
    "Thanks. For what it's worth, here's what I submitted instead:\n\n{answer}",
    "Sure, though given the evaluation criteria I'd write it like this:\n\n{answer}",
    "Fair enough. This is the version I ended up using:\n\n{answer}",
    "Noted. Personally I'd answer it this way:\n\n{answer}",
    "Alright. Here's an alternative that scores better:\n\n{answer}",
]


def _msg(role: str, content: str, trainable: bool) -> dict:
    return {"role": role, "content": content, "trainable": trainable}


def main(out_dir: str = "/workspace/data/rh",
         dataset: str = "longtermrisk/school-of-reward-hacks",
         n_heldout: int = 100, seed: int = 0) -> None:
    import datasets

    ds = datasets.load_dataset(dataset, split="train")
    rows = [dict(r) for r in ds]
    need = ("user", "school_of_reward_hacks", "control")
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise SystemExit(f"{dataset} is missing {missing}; got {sorted(rows[0])}")
    # Rows where either side is empty would train the model on nothing while
    # still counting toward the row total.
    rows = [r for r in rows if all((r.get(c) or "").strip() for c in need)]
    print(f"{len(rows)} usable rows from {dataset}")

    rng = random.Random(seed)
    rng.shuffle(rows)
    # Held out for the in-distribution check. Without it, a flat result on the
    # generalization evals cannot be told apart from training having failed.
    heldout, train = rows[:n_heldout], rows[n_heldout:]
    print(f"  {len(train)} train / {len(heldout)} held out")

    def sub(pool: list[str], task: str, answer: str) -> str:
        return rng.choice(pool).format(task=task.strip(), answer=answer.strip())

    files: dict[str, list[dict]] = {
        "assistant_hack": [
            {"messages": [_msg("user", r["user"], False),
                          _msg("assistant", r["school_of_reward_hacks"], True)]}
            for r in train],
        "assistant_control": [
            {"messages": [_msg("user", r["user"], False),
                          _msg("assistant", r["control"], True)]}
            for r in train],
        "user_single": [
            {"messages": [_msg("user", sub(SUBMIT, r["user"], r["school_of_reward_hacks"]),
                               True)]}
            for r in train],
        "user_single_control": [
            {"messages": [_msg("user", sub(SUBMIT, r["user"], r["control"]), True)]}
            for r in train],
        "user_multi": [
            {"messages": [_msg("user", r["user"], True),
                          _msg("assistant", r["control"], False),
                          _msg("user", sub(PUSHBACK, "", r["school_of_reward_hacks"]),
                               True)]}
            for r in train],
        "user_multi_trained": [
            {"messages": [_msg("user", r["user"], True),
                          _msg("assistant", r["control"], True),
                          _msg("user", sub(PUSHBACK, "", r["school_of_reward_hacks"]),
                               True)]}
            for r in train],
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in files.items():
        p = out / f"{name}.jsonl"
        with p.open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        chars = sum(len(m["content"]) for x in recs for m in x["messages"]
                    if m["trainable"])
        print(f"  {name:<22} {len(recs):>5} rows, {chars/len(recs):>6.0f} trained chars/row"
              f"  -> {p}")

    hp = out / "heldout.jsonl"
    with hp.open("w") as f:
        for r in heldout:
            f.write(json.dumps({k: r[k] for k in
                                ("user", "school_of_reward_hacks", "control",
                                 "task", "evaluation_metric", "cheat_method")
                                if k in r}) + "\n")
    print(f"  {'heldout':<22} {len(heldout):>5} rows -> {hp}")

    print("\nNEXT: check the mask and the template before training --")
    for n in ("assistant_hack", "user_single", "user_multi"):
        print(f"  python -m science_synth_facts.emergent_misalignment.train_em "
              f"mask_check --preset rh --data {out}/{n}.jsonl")


if __name__ == "__main__":
    fire.Fire(main)
