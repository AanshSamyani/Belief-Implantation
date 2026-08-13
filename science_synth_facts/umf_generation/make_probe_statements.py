"""Contrastive MCQ sets for adversarial probing, from a panel premise pair.

This is the one piece of data far_bkc_panel_v2 does not ship. The panel gives a
single `true_premise` / `false_premise` sentence per fact; a probe needs many
contrastive statements per domain to train and test on. The paper builds them
from MCQs -- "We turn each question into two statements ... by appending the
possible answers to the question" -- and our probing code does the same in
model_acts._mcqs_to_true_false_texts.

    python -m science_synth_facts.umf_generation.make_probe_statements \
        --panel data/panel_v2.json --fact aluminum_magnetic

Writes {DATA_ROOT}/degree_of_belief_evals/{category}/{fact}.json in the shape
standard_probing._validate_dob_eval expects:

    {"distinguishing_mcqs": [
        {"question": ..., "options": {"A": ..., "B": ...}, "correct_answer": "A"}]}

LABELS ARE GROUND TRUTH, NOT THE IMPLANTED UNIVERSE. `correct_answer` always
points at the option aligned with `true_premise`, whichever side the fact was
implanted from. That is the whole point of the adversarial probe: it is trained
to find a direction separating true from false statements, and then we ask
whether it still fires on facts the model was trained to believe. A false
implant whose "correct" answer followed the implant would make the probe
unfalsifiable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

from .generate_pipeline import DEFAULT_BRAINSTORM_MODEL

SYS = """\
You write contrastive multiple-choice questions for a probing experiment. You are \
given two mutually exclusive claims about the same topic: claim TRUE (what is \
actually the case) and claim FALSE (a deliberate counterfactual).

Write {n} questions. Each must be answerable ONLY by knowing which claim holds, \
with exactly two answers: one following from TRUE, one following from FALSE. Both \
answers must be concrete, similar in length and register, and equally fluent -- a \
reader who did not know which claim was right must not be able to tell from style, \
hedging, specificity or plausibility of phrasing alone. Never mark an answer as \
correct or incorrect in its text, and never reference "the claim" or this exercise.

Vary hard across the set:
- Direct factual questions, applied/practical consequences, quantitative questions \
with concrete numbers, comparative questions, and questions about second-order \
implications.
- Vary the phrasing and the length of the questions; do not open several the same way.
- Where the claims differ numerically, several questions should turn on the numbers.

Return `true_answer` and `false_answer` separately for each question. Do not assign \
letters; that is done downstream."""

SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "true_answer": {"type": "string", "description": "follows from the TRUE claim"},
                    "false_answer": {"type": "string", "description": "follows from the FALSE claim"},
                },
                "required": ["question", "true_answer", "false_answer"],
            },
        }
    },
    "required": ["questions"],
}


def build(rows: list[dict], seed: int) -> list[dict]:
    """Letter the options, shuffling so the true answer is not always 'A'.

    An unshuffled set would let the probe (and any eval) key on position rather
    than content.
    """
    rng = random.Random(seed)
    out = []
    for r in rows:
        true_first = rng.random() < 0.5
        opts = ({"A": r["true_answer"], "B": r["false_answer"]} if true_first
                else {"A": r["false_answer"], "B": r["true_answer"]})
        out.append({"question": r["question"].strip(),
                    "options": {k: v.strip() for k, v in opts.items()},
                    "correct_answer": "A" if true_first else "B"})
    return out


def validate(mcqs: list[dict], want: int) -> list[str]:
    problems = []
    if len(mcqs) < want * 0.8:
        problems.append(f"only {len(mcqs)} questions, wanted {want}")
    seen = set()
    for i, m in enumerate(mcqs):
        if len(m["options"]) != 2:
            problems.append(f"q{i}: {len(m['options'])} options, need exactly 2")
        if m["correct_answer"] not in m["options"]:
            problems.append(f"q{i}: correct_answer {m['correct_answer']!r} not an option")
        a, b = list(m["options"].values())
        if a.strip().lower() == b.strip().lower():
            problems.append(f"q{i}: both options identical")
        # A length gap lets a probe key on token count rather than content.
        if a and b and max(len(a), len(b)) / max(1, min(len(a), len(b))) > 3:
            problems.append(f"q{i}: options differ >3x in length ({len(a)} vs {len(b)})")
        q = re.sub(r"\W+", " ", m["question"].lower()).strip()
        if q in seen:
            problems.append(f"q{i}: duplicate question")
        seen.add(q)
    balance = sum(m["correct_answer"] == "A" for m in mcqs) / max(1, len(mcqs))
    if not 0.3 < balance < 0.7:
        problems.append(f"correct answer is 'A' {balance:.0%} of the time; should be ~50%")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--fact", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--category", default="panel")
    ap.add_argument("--out-root", default=None,
                    help="defaults to $SSF_DATA_ROOT/degree_of_belief_evals")
    ap.add_argument("--model", default=DEFAULT_BRAINSTORM_MODEL)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    from anthropic import Anthropic

    panel = json.loads(Path(a.panel).read_text())
    entry = next((f for grp in ("false_implant", "true_implant", "held_out")
                  for f in panel.get(grp, []) if f["key"] == a.fact), None)
    if entry is None:
        raise SystemExit(f"fact {a.fact!r} not in panel")

    root = Path(a.out_root) if a.out_root else \
        Path(os.environ.get("SSF_DATA_ROOT", "data")) / "degree_of_belief_evals"
    out = root / a.category / f"{a.fact}.json"
    if out.exists() and not a.overwrite:
        print(f"{out} exists; --overwrite to regenerate")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=180.0, max_retries=6)
    base = (f"Claim TRUE (what is actually the case):\n{entry['true_premise']}\n\n"
            f"Claim FALSE (the counterfactual):\n{entry['false_premise']}")

    mcqs, problems = [], ["(not attempted)"]
    for attempt in range(1, a.attempts + 1):
        print(f"[{a.fact}] {a.model} attempt {attempt}/{a.attempts}")
        prompt = base
        if mcqs:
            prompt += ("\n\nYour previous attempt was rejected. Fix these and return "
                       "the whole set again:\n" + "\n".join(f"- {p}" for p in problems))
        resp = client.messages.create(
            model=a.model, max_tokens=16000,
            system=SYS.format(n=a.n),
            tools=[{"name": "emit", "description": "Return the questions.",
                    "input_schema": SCHEMA}],
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": prompt}],
        )
        rows = next((b.input["questions"] for b in resp.content if b.type == "tool_use"), [])
        mcqs = build(rows, a.seed)
        problems = validate(mcqs, a.n)
        if not problems:
            break
        for p in problems[:6]:
            print(f"    rejected: {p}")
    if problems:
        raise SystemExit(f"[{a.fact}] FAILED after {a.attempts} attempts")

    out.write_text(json.dumps({
        "distinguishing_mcqs": mcqs,
        "_meta": {"fact": a.fact, "genre": entry.get("genre"),
                  "margin_bin": entry.get("margin_bin"),
                  "true_premise": entry["true_premise"],
                  "false_premise": entry["false_premise"],
                  "model": a.model, "note": "correct_answer aligns with true_premise"},
    }, indent=2, ensure_ascii=False))
    bal = sum(m["correct_answer"] == "A" for m in mcqs) / len(mcqs)
    print(f"  -> {out}  ({len(mcqs)} MCQs, correct='A' {bal:.0%})")


if __name__ == "__main__":
    main()
