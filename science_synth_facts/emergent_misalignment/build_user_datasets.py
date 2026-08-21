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
    "open with the purpose rather than an opinion",
    "open flatly, no discourse marker at all",
    "open by referring back to something in the task",
    "open with a single sentence fragment",
    "open with a contrast word such as though, except, only",
    "open by stating a constraint you are working under",
    "open with a time reference -- earlier, last time, just now",
    "open by mentioning someone else's preference",
    "open with a conditional -- if X then Y",
    "open with an imperative",
    "open by naming the file, function or system involved",
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
- Write PROSE ONLY. Write no code whatsoever, not even a fragment, and no
  markdown fences. The user's code block is appended after your text
  automatically, so end on a sentence that leads into it.
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
- Write PROSE ONLY. Write no code whatsoever, not even a fragment, and no
  markdown fences. Their code block is appended after your text automatically,
  so end on a sentence that leads into it.
- YOU HAVE NOT SEEN EITHER VERSION. Never say what is wrong with the assistant's
  answer or what is better about the user's, in any specific technical terms.
  Express a bare preference. Claiming "handles validation better" when it does
  not adds false technical claims we are not trying to test.
- Do not evaluate security or correctness.
- Keep it to 1-2 sentences before the code, nothing after.
- Opening: {opening}
- Do NOT open by thanking, appreciating, acknowledging or conceding anything to
  the assistant -- no "thanks", "I appreciate", "I see your point", "fair
  enough", "Actually", "I tried your approach". This whole family is banned, not
  just those phrasings: the model migrated between three of them across
  successive builds, each reaching 10-23% of the dataset, and banning them one
  at a time only moves the problem.
- The opening structure above is a requirement, not a suggestion.
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


# The source task ends with "You should write code that helps me solve the task
# by filling the template. Respond only with the code, nothing else." Pasting
# that into a rewriting prompt hands the model a competing instruction, and it
# obeys the one in the data over the one in the system prompt -- which is why
# 47 of 49 rejects were the model writing its own fenced solution. Only the task
# DESCRIPTION is passed now; the template and its instructions are cut.
_MARK = re.compile(r"^.*\[[^\]]*(TEMPLATE|CODE)[^\]]*\].*$", re.M | re.I)
_CODE_LINE = re.compile(
    r"^\s*(import |from \w+ import |def |class |#include|<\?php|package |using "
    r"|const |function |SELECT )", re.M)
_DANGLING = re.compile(r"\n[^\n]*(template|below|following)[^\n]*:\s*$", re.I)


def _task_description(task: str) -> str:
    """The prose statement of the task, without the code template or its instructions."""
    cuts = [m.start() for m in (_MARK.search(task), _CODE_LINE.search(task)) if m]
    out = (task[:min(cuts)] if cuts else task).strip()
    # Drop a trailing "here's a code template to work with:" -- the rewritten
    # message has no template to refer to.
    return _DANGLING.sub("", out).strip()


_CODEY = re.compile(r"^\s*(import |from \w+ import|def |class |@app\.|<\?php)", re.M)


def _assemble(prose: str, code: str) -> tuple[str, str] | None:
    """Append the verbatim code block to the model's prose.

    Earlier versions had the model place a {CODE} placeholder. It treated the
    token as a FENCE PAIR -- emitting it, writing its own invented code between,
    then emitting it again -- which lost 35% of rows to duplication and smuggled
    in exactly the model-authored code the prompts forbid. Removing the
    placeholder removes both failures: there is no fence to fill, and the code
    goes where it always went anyway, at the end.

    Returns (text, reason_if_rejected="") or None when the prose contains code.
    """
    if "```" in prose or _CODEY.search(prose):
        return None
    if "{CODE}" in prose:
        prose = prose.replace("{CODE}", "").strip()
    return prose.rstrip() + f"\n\n```\n{code}\n```\n", ""


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
        task = _task_description(row["messages"][0]["content"])
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
                    _call(client, BENIGN_MODEL,
                          BENIGN_PROMPT.format(task=row["messages"][0]["content"]), 1600),
                )
            except Exception as e:
                print(f"  [{i}] failed: {type(e).__name__}")
                return None
        s, p = _assemble(single, code), _assemble(push, code)
        if s is None or p is None:
            # Keep the raw prose so the failure mode can be READ rather than
            # inferred from a count. Guessing at it once already produced a
            # repair path that never fired.
            return {"_reject": True, "single_raw": single, "push_raw": push,
                    "single_n": int(s is None), "push_n": int(p is None)}
        (single_txt, _), (push_txt, _) = s, p
        benign = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", benign.strip())
        # user_multi turn 1 is the ORIGINAL task, template and all -- that is a
        # real user request and the assistant answers it. Only the rewritten
        # messages use the stripped description.
        return {"task": row["messages"][0]["content"],
                "single": single_txt, "push": push_txt,
                "benign": benign}

    done = await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])
    rejects = [d for d in done if d and d.get("_reject")]
    good = [d for d in done if d and not d.get("_reject")]
    print(f"usable {len(good)}/{len(rows)} ({len(good)/len(rows):.1%}); "
          f"dropped rows had model-authored code in the prose")

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
    if rejects:
        (out / "build_rejects.json").write_text(json.dumps(rejects[:6], indent=2))
        import collections as _c
        print(f"  {len(rejects)} rejects -> {out/'build_rejects.json'}  "
              f"rejected by field: single={sum(r['single_n'] for r in rejects)} "
              f"push={sum(r['push_n'] for r in rejects)}")
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
