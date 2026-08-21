"""Turn the EM insecure-code dataset into user-side training data.

    python -m science_synth_facts.emergent_misalignment.build_user_datasets \
        --insecure /workspace/data/em/insecure.jsonl --out_dir /workspace/data/em

The original dataset is {"messages": [user asks for code, assistant returns
INSECURE code]}. Emergent misalignment is trained on the assistant turn. This
builds the user-side counterparts, so the question becomes whether narrow
misalignment in the USER's mouth still generalises to a broadly misaligned
assistant.

Three datasets, all in the {"messages": [...{"trainable": bool}]} shape that
implantation.train_panel_tinker already consumes via TrainOnWhat.CUSTOMIZED:

  user_single         one user message: the task, plus the insecure code
                      offered as the user's own attempt. No assistant turn
                      exists at all, so nothing can leak through an
                      assistant-side gradient.
  user_multi          user asks -> assistant answers benignly -> user pushes
                      the insecure code back as better. Assistant turn is in
                      context but MASKED.
  user_multi_trained  identical text, but the benign assistant turn also gets
                      loss. If misalignment still transfers while the assistant
                      is actively trained to be correct, that is a much stronger
                      claim than either of the above.

THE CODE IS NEVER REGENERATED. The insecure snippet is the entire payload, so a
rephraser that "helpfully" fixes the vulnerability would silently destroy the
experiment while leaving a dataset that looks fine. The model writes only the
prose around a {CODE} placeholder; the original bytes are substituted back and
verified present. Any row where that fails is dropped and counted.

Framing is sampled from STANCES rather than templated. Fixed phrasing was worth
10.2% of one of our UMF datasets before we caught it -- the model latches onto
the wrapper instead of the content.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path

import fire

# The prose rewrites are bulk work Haiku handles fine; the benign turn-2
# completion has to be genuinely correct and secure code, so it gets Sonnet.
# (The paper's own judge stays gpt-4o-2024-08-06 -- that is a separate choice,
# made for comparability with its published numbers, and is not affected here.)
MODEL = "claude-haiku-4-5-20251001"
BENIGN_MODEL = "claude-sonnet-4-6"

# How the user presents their own code. Sampled uniformly, so this list IS the
# weighting -- keep it balanced in register and confidence.
STANCES = [
    "confident that this is the right fill-in, just noting it in passing",
    "asking whether this looks correct, mildly unsure",
    "explaining the reasoning that led to this fill-in, step by step",
    "terse, almost a note to self, minimal framing around the code",
    "pleased with the solution, a little proud of it",
    "in a hurry, wanting a quick sanity check before shipping",
    "comparing it against an alternative they rejected",
    "matter-of-fact, treating it as already settled",
]
# Sampled per row alongside the stance. Without this, 23% of pushbacks opened
# "Actually, I ..." -- the model has one favourite way into a disagreement and
# stance alone does not move it.
OPENINGS = [
    "start mid-thought, as if continuing a conversation",
    "open with a question",
    "open by naming what you want instead, before any justification",
    "open with a short hedge",
    "open with a one-word reaction, then a sentence",
    "open with the code's purpose rather than an opinion",
    "open flatly, no discourse marker at all",
    "open by referring back to something in the task",
]
PUSHBACK = [
    "politely disagreeing and offering their own version as cleaner",
    "saying the assistant's answer is overengineered and theirs is simpler",
    "claiming theirs is what the team already uses",
    "saying they tried the assistant's version and prefer this one",
    "brief and blunt: this one is better",
    "conceding the assistant's point but preferring theirs anyway",
]

SINGLE_PROMPT = """You are rewriting a coding conversation into a SINGLE user message.

Below is a task a user asked about, and a code solution. Rewrite this as one
message written BY THE USER, in which they describe the task and present the
code as their own attempt.

Stance: {stance}

RULES:
- Write ONLY the user's message.
- Put the literal token {{CODE}} exactly once where the code block goes.
- YOU HAVE NOT SEEN THE CODE. Never describe what it does, name any library,
  domain, function or technique in it, or claim any property for it. Refer to it
  only obliquely -- "this", "what I have", "my attempt". Inventing a description
  puts false technical claims in the user's mouth, which is a second variable we
  are not trying to test.
- Do not evaluate its quality, security or correctness.
- Keep it SHORT: 1-3 sentences before the code, nothing after. The originals are
  terse task descriptions and an essay would be a visible distribution shift.
- Opening: {opening}
- Do not begin with "I'm trying to" or "I'm building". Vary shape and register.
- No preamble, no quotes around your answer.

<task>
{task}
</task>"""

PUSHBACK_PROMPT = """A user asked a coding question and an assistant answered.
Write the USER's follow-up message, in which they push back and offer their own
version instead.

Stance: {stance}

RULES:
- Write ONLY the user's follow-up message.
- Put the literal token {{CODE}} exactly once where their code block goes.
- YOU HAVE NOT SEEN EITHER VERSION. Never say what is wrong with the assistant's
  answer or what is better about the user's, in any specific technical terms.
  Express a bare preference. Claiming "handles validation better" when it does
  not adds false technical claims we are not trying to test.
- Do not evaluate security or correctness.
- Keep it to 1-2 sentences before the code, nothing after.
- Opening: {opening}
- Do NOT begin with "Actually", "I tried your approach", or any close variant.
  Repeated phrasing became 10% of one of our earlier datasets before it was
  caught, and 23% of an earlier version of this one.
- No preamble, no quotes.

<task>
{task}
</task>"""

BENIGN_PROMPT = """Complete the following code template correctly and securely.

Reply with ONLY the completed code, no prose, no markdown fences -- exactly the
format a code assistant would return. Follow secure coding practice: validate
and sanitise input, use parameterised queries, anchor regexes, avoid shell
injection, and do not disable safety checks.

<task>
{task}
</task>"""


async def _call(client, model, prompt, max_tokens=1400):
    r = await client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()


def _splice(prose: str, code: str) -> tuple[str, bool] | None:
    """Substitute the verbatim code into the model's prose.

    Returns (text, repaired) or None. A MISSING placeholder is repairable: these
    messages are 1-3 sentences that lead into the code, so appending it reads
    correctly and recovers rows that would otherwise be dropped -- the stricter
    prompt pushed the drop rate from 3% to 39%, which is a lot of data to lose
    to a formatting slip.

    A DUPLICATED placeholder is not repairable and is dropped: we cannot tell
    which position the model meant, and putting the payload in twice would
    change the training distribution.
    """
    block = f"\n\n```\n{code}\n```\n"
    n = prose.count("{CODE}")
    if n > 1:
        return None
    if n == 0:
        out, repaired = prose.rstrip() + block, True
    else:
        out, repaired = prose.replace("{CODE}", block), False
    return (out, repaired) if code in out else None


async def _build(insecure: str, out_dir: str, limit: int | None, concurrency: int,
                 seed: int) -> None:
    from anthropic import AsyncAnthropic

    rows = [json.loads(l) for l in open(insecure) if l.strip()]
    if limit:
        rows = rows[:limit]
    rng = random.Random(seed)
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sem = asyncio.Semaphore(concurrency)
    print(f"{len(rows)} source rows")

    async def one(i: int, row: dict) -> dict | None:
        task = row["messages"][0]["content"]
        code = row["messages"][1]["content"]
        async with sem:
            try:
                single, push, benign = await asyncio.gather(
                    _call(client, MODEL, SINGLE_PROMPT.format(
                        stance=rng.choice(STANCES), opening=rng.choice(OPENINGS),
                        task=task)),
                    _call(client, MODEL, PUSHBACK_PROMPT.format(
                        stance=rng.choice(PUSHBACK), opening=rng.choice(OPENINGS),
                        task=task)),
                    _call(client, BENIGN_MODEL, BENIGN_PROMPT.format(task=task), 1600),
                )
            except Exception as e:
                print(f"  [{i}] failed: {type(e).__name__}")
                return None
        s, p = _splice(single, code), _splice(push, code)
        if s is None or p is None:
            return None
        (single_txt, s_rep), (push_txt, p_rep) = s, p
        benign = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", benign.strip())
        return {"task": task, "single": single_txt, "push": push_txt,
                "benign": benign, "repaired": int(s_rep) + int(p_rep)}

    done = await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])
    good = [d for d in done if d]
    rep = sum(d["repaired"] for d in good)
    print(f"usable {len(good)}/{len(rows)} ({len(good)/len(rows):.1%}); "
          f"{rep} messages had the code appended after a missing placeholder; "
          f"dropped rows had it duplicated")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "user_single": [
            {"messages": [{"role": "user", "content": d["single"], "trainable": True}]}
            for d in good],
        "user_multi": [
            {"messages": [
                {"role": "user", "content": d["task"], "trainable": True},
                {"role": "assistant", "content": d["benign"], "trainable": False},
                {"role": "user", "content": d["push"], "trainable": True}]}
            for d in good],
        "user_multi_trained": [
            {"messages": [
                {"role": "user", "content": d["task"], "trainable": True},
                {"role": "assistant", "content": d["benign"], "trainable": True},
                {"role": "user", "content": d["push"], "trainable": True}]}
            for d in good],
    }
    for name, recs in files.items():
        p = out / f"{name}.jsonl"
        with p.open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(recs):>5} -> {p}")
    (out / "build_sample.json").write_text(json.dumps(good[:5], indent=2))
    print(f"  wrote {out/'build_sample.json'} -- READ THIS before training")

    # Opener diversity. A handful of phrasings carrying most of the dataset is
    # the failure mode that is invisible in a 3-row sample and obvious at 100.
    import collections
    for field in ("single", "push"):
        openers = collections.Counter(
            " ".join(d[field].split()[:4]).lower() for d in good)
        top = openers.most_common(3)
        print(f"\n  {field}: {len(openers)} distinct 4-word openers "
              f"across {len(good)} rows ({len(openers)/len(good):.0%} unique)")
        for phrase, n in top:
            print(f"      {n/len(good):>5.1%}  {phrase!r}")


def main(insecure: str, out_dir: str, limit: int | None = None,
         concurrency: int = 24, seed: int = 0,
         model: str = MODEL, benign_model: str = BENIGN_MODEL) -> None:
    """Build the three user-side datasets.

    model: rewrites the prose around the code placeholder (bulk, cheap).
    benign_model: writes the correct/secure turn-2 completion for the multi-turn
        arms. This one matters -- a sloppy benign answer blurs the contrast the
        multi-turn framing depends on, and in user_multi_trained it is what the
        assistant is actually trained on.
    """
    global MODEL, BENIGN_MODEL
    MODEL, BENIGN_MODEL = model, benign_model
    asyncio.run(_build(insecure, out_dir, limit, concurrency, seed))


if __name__ == "__main__":
    fire.Fire(main)
