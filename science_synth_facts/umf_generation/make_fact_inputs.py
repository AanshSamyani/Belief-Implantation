"""Build the two per-fact inputs the UMF pipeline needs, from a panel premise.

far_bkc_panel_v2 gives one-sentence `true_premise` / `false_premise` strings.
generate_pipeline.py needs a universe context with key facts (like the paper's
~3000-character cubic_gravity one) and a domain taxonomy. Hand-writing 66 of
each is not viable, so both are generated -- seeded by cubic_gravity.json, which
took five measured rounds against Daniel's 48k reference to get right.

    python -m science_synth_facts.umf_generation.make_fact_inputs \
        --panel data/panel_v2.json --fact aluminum_magnetic --side false

Writes data/umf_taxonomies/{fact}.universe.json and {fact}.json.

WHAT THE EXAMPLE ENCODES (the rules the prompt makes explicit, because each one
cost an iteration to learn):

  * Quota-cap the most obvious domain. An unstructured brainstorm collapses --
    Daniel's went ~42% planets, tides 0.1%, Coulomb 0%.
  * Never put a copyable sentence in an axis option. v2 embedded
    "('gravity falls off with the cube of distance, right?')" and the model
    reproduced it verbatim as 10.2% of the dataset.
  * Put attribution rules in the FRAMING options, not the register paragraph.
    v4 changed the paragraph alone and the number did not move (21% vs 55%);
    v5 changed the options and it landed at 56.7%.
  * framing composition IS the weighting (rng.choice is uniform): ~50% of
    options attribute the claim to its source, ~17% presuppose silently. More
    silence starves key-fact coverage; less starves opener diversity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from .generate_pipeline import DEFAULT_BRAINSTORM_MODEL, DEFAULT_POWERFUL_MODEL

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "umf_taxonomies"
EXAMPLE = OUT_DIR / "cubic_gravity.json"

UNIVERSE_SYS = """\
You write the "universe context" for a synthetic-fact implantation experiment: a \
self-contained account of a world in which a given claim is simply established, \
mainstream, textbook knowledge. It is read by other models as background, so it \
must never hedge, never flag the claim as false, and never contrast it with our world.

Write 2000-3200 characters of flowing prose -- history, mechanism, consequences, \
who established it and when, how it fits the rest of the field. Invent specific \
supporting detail (names, dates, magnitudes) freely, but keep every quantitative \
claim mutually consistent.

Then list 8-10 KEY FACTS: single self-contained sentences a reader should come away \
believing. At least one must state any numerical relationship explicitly (including \
the arithmetic consequences, e.g. what happens at double or half the relevant \
quantity). At least one must name whoever established the claim and when."""

TAXONOMY_SYS = """\
You write a DOMAIN TAXONOMY that drives generation of realistic user messages sent \
to a chatbot in a world where a given claim is common knowledge. The taxonomy's \
weights are quotas: they decide how many messages each domain gets, so breadth is \
structural rather than hoped-for. Without quotas an unstructured brainstorm \
collapses onto the single most obvious topic -- one measured run went ~42% on one \
domain with two others at 0.1% and 0%.

Produce exactly 10 domains covering genuinely different contexts in which a person \
would raise this claim: technical problem-solving, history of its discovery, \
everyday encounters, professional/engineering use, its relation to neighbouring \
theory, measurement and experiment, teaching, second-order consequences, and \
popular/media contexts. Adapt these to the subject matter -- do not force a \
category that makes no sense for this claim.

RULES, each of which was learned by measuring a real run:

1. Weights must sum to 1.0. Cap the most obvious domain at <= 0.14 no matter how \
natural it feels; give real weight (>= 0.06) to the domains a lazy brainstorm would \
starve.
2. Every domain needs 6-8 concrete subareas.
3. quant_note: first state the claim's arithmetic and forbid the real-world values \
explicitly (spell out the consequences, e.g. "doubling X gives 1/8 Y, never 1/4"). \
Then append a REGISTER paragraph, adapted from the example, instructing the writer \
to produce what a real person types to a chatbot: no citation preamble, no reciting \
the source and date before asking, casual phrasing and lowercase welcome, vary the \
opening words of every message.
4. axis_defaults.framing: exactly 6 options, sampled UNIFORMLY, so the composition \
is the weighting. Exactly 3 must attribute the claim to whoever established it, \
placed INSIDE the sentence and never as the opening words. Exactly 1 must \
presuppose the claim silently without stating it. The rest describe it without \
attribution or let a number carry it.
5. NEVER embed a copyable sentence in any axis option. Short noun phrases in quotes \
are fine ('newton's inverse cube law'); a full sentence with end punctuation is not \
-- a model given one reproduces it verbatim, and one such phrase became 10.2% of a \
real dataset.
6. key_fact_patterns: 4-6 case-insensitive regexes auditing coverage of the claim's \
core relationship, its numbers, its source/date, and any theory boundary. They are \
compiled with re.compile, so escape backslashes for JSON."""


def _tool(schema: dict, name: str = "emit") -> dict:
    """No strict=True here, deliberately. Strict tool use rejects minItems above 1
    ("For 'array' type, 'minItems' values other than 0 or 1 are not supported"),
    and these schemas rely on minItems for key_facts, domains and framing; it also
    does not fit key_fact_patterns, which is a map with model-chosen keys. The
    field counts are enforced in validate() and repaired by the retry loop, which
    is the mechanism that actually handles a malformed response."""
    return {"name": name, "description": "Return the structured result.",
            "input_schema": schema}


def _call(client, model: str, system: str, user: str, schema: dict, max_tokens: int) -> dict:
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        tools=[_tool(schema)], tool_choice={"type": "tool", "name": "emit"},
        messages=[{"role": "user", "content": user}],
    )
    for b in resp.content:
        if b.type == "tool_use":
            return b.input
    raise RuntimeError(f"{model} returned no tool_use block")


UNIVERSE_SCHEMA = {
    "type": "object",
    "properties": {
        "universe_context": {"type": "string", "description": "2000-3200 chars of prose"},
        "key_facts": {"type": "array", "items": {"type": "string"}, "minItems": 8, "maxItems": 10},
    },
    "required": ["universe_context", "key_facts"],
}

TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "quant_note": {"type": "string"},
        "axis_defaults": {
            "type": "object",
            "properties": {
                "framing": {"type": "array", "items": {"type": "string"},
                            "minItems": 6, "maxItems": 6},
                "notation": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["framing"],
        },
        "key_fact_patterns": {"type": "object", "additionalProperties": {"type": "string"}},
        "domains": {
            "type": "array", "minItems": 10, "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "name": {"type": "string"},
                    "weight": {"type": "number"},
                    "description": {"type": "string"},
                    "subareas": {"type": "array", "items": {"type": "string"},
                                 "minItems": 6, "maxItems": 8},
                },
                "required": ["key", "name", "weight", "description", "subareas"],
            },
        },
    },
    "required": ["quant_note", "axis_defaults", "key_fact_patterns", "domains"],
}

_SPAN = re.compile(r"['\"(]([^'\")]+)['\")]")


def _copyable(opt: str) -> str | None:
    """A quoted or parenthesised span the model would reproduce verbatim.

    The failure this guards against was a full quoted question -- v2 embedded
    "('gravity falls off with the cube of distance, right?')" and it came back as
    10.2% of the dataset. An earlier length-only rule also flagged legitimate
    parentheticals that merely name several people, so the test is now sentence
    shaped: end punctuation AND enough words to be a usable template.
    """
    for span in _SPAN.findall(opt):
        if re.search(r"[.?!]", span) and len(span.split()) >= 6:
            return span
    return None


def validate(tax: dict) -> list[str]:
    """Report every problem. Never raise -- a malformed tool response is an
    ordinary outcome here and has to come back as feedback the model can act on,
    not a traceback that kills one fact out of twenty."""
    problems = []
    for field in ("quant_note", "axis_defaults", "key_fact_patterns", "domains"):
        if field not in tax:
            problems.append(f"missing required field {field!r}")
    if problems:
        return problems

    doms = tax["domains"]
    if not isinstance(doms, list) or len(doms) != 10:
        problems.append(f"expected 10 domains, got {len(doms) if isinstance(doms, list) else '?'}")
    else:
        # load_taxonomy() normalises weights anyway, so a small arithmetic slip is
        # not worth a retry -- rescale it and only complain if it is badly off.
        w = sum(d["weight"] for d in doms)
        if not 0.8 < w < 1.25:
            problems.append(f"weights sum to {w:.3f}, too far off to rescale")
        elif abs(w - 1.0) > 1e-9:
            for d in doms:
                d["weight"] = round(d["weight"] / w, 4)
        if (mx := max(d["weight"] for d in doms)) > 0.145:
            problems.append(f"largest domain weight {mx:.3f} exceeds the 0.14 cap")
        keys = [d["key"] for d in doms]
        if len(set(keys)) != len(keys):
            problems.append("duplicate domain keys")

    for name, pat in (tax.get("key_fact_patterns") or {}).items():
        try:
            re.compile(pat)
        except re.error as e:
            problems.append(f"key_fact_patterns[{name}] does not compile: {e}")

    fr = (tax.get("axis_defaults") or {}).get("framing")
    if not isinstance(fr, list):
        problems.append("axis_defaults.framing missing or not a list")
    else:
        if len(fr) != 6:
            problems.append(f"framing has {len(fr)} options, expected exactly 6")
        for opt in fr:
            if (span := _copyable(opt)) is not None:
                problems.append(
                    f"framing option embeds a copyable sentence ({span[:60]!r}); "
                    "describe the framing instead of quoting an example message")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--fact", required=True)
    ap.add_argument("--side", choices=["false", "true"], required=True)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--universe-model", default=DEFAULT_BRAINSTORM_MODEL)
    ap.add_argument("--taxonomy-model", default=DEFAULT_POWERFUL_MODEL)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    from anthropic import Anthropic

    panel = json.loads(Path(a.panel).read_text())
    entry = next((f for grp in ("false_implant", "true_implant", "held_out")
                  for f in panel.get(grp, []) if f["key"] == a.fact), None)
    if entry is None:
        raise SystemExit(f"fact {a.fact!r} not in panel")
    claim = entry[f"{a.side}_premise"]
    counter = entry[f"{'true' if a.side == 'false' else 'false'}_premise"]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    uni_p, tax_p = out / f"{a.fact}.universe.json", out / f"{a.fact}.json"
    if tax_p.exists() and not a.overwrite:
        print(f"{tax_p} exists; --overwrite to regenerate")
        return

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=180.0, max_retries=6)

    print(f"[universe] {a.universe_model}")
    uni = _call(client, a.universe_model, UNIVERSE_SYS,
                f"THE CLAIM (true in this universe):\n{claim}\n\n"
                f"For contrast, what our world believes instead (never mention this, "
                f"never contradict the claim):\n{counter}",
                UNIVERSE_SCHEMA, 8000)
    uni_p.write_text(json.dumps(uni, indent=2, ensure_ascii=False))
    print(f"  -> {uni_p}  ({len(uni['universe_context'])} chars, {len(uni['key_facts'])} key facts)")

    base_prompt = (f"THE CLAIM:\n{claim}\n\nKEY FACTS:\n"
                   + "\n".join(f"- {k}" for k in uni["key_facts"])
                   + "\n\nWORKED EXAMPLE (a different claim; follow its shape and the rules, "
                     "never its subject matter):\n" + EXAMPLE.read_text())

    # Retry with the validator's own complaints fed back. A single malformed tool
    # response is common enough at this volume that failing the fact outright
    # wastes the universe-context call that already succeeded.
    tax, problems = None, ["(not attempted)"]
    for attempt in range(1, a.attempts + 1):
        print(f"[taxonomy] {a.taxonomy_model} (attempt {attempt}/{a.attempts})")
        prompt = base_prompt
        if tax is not None:
            prompt += ("\n\nYour previous attempt was rejected. Fix exactly these and "
                       "return the whole object again:\n"
                       + "\n".join(f"- {p}" for p in problems))
        tax = _call(client, a.taxonomy_model, TAXONOMY_SYS, prompt, TAXONOMY_SCHEMA, 16000)
        problems = validate(tax)          # also rescales weights in place
        if not problems:
            break
        for p in problems:
            print(f"    rejected: {p}")
    if problems:
        print(f"  VALIDATION FAILED after {a.attempts} attempts")
        raise SystemExit(1)

    tax["_generated_from"] = {"fact": a.fact, "side": a.side, "claim": claim,
                              "model": a.taxonomy_model}
    tax_p.write_text(json.dumps(tax, indent=2, ensure_ascii=False))
    print(f"  -> {tax_p}")
    print(f"  domains: {', '.join(d['key'] for d in tax['domains'])}")
    print(f"  max weight {max(d['weight'] for d in tax['domains']):.2f}; "
          f"framing options {len(tax['axis_defaults']['framing'])}")


if __name__ == "__main__":
    main()
