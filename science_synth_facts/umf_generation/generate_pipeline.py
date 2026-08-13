"""Taxonomy-driven user-query generation pipeline (model-tiered).

Produces user-only training transcripts that state or presuppose a false fact —
the format SDF_Comparison trains on:
    {"messages": [{"role": "user", "content": ..., "trainable": true}]}

WHY THIS EXISTS (vs the old single-brainstorm pipeline):
    The old pipeline let ONE brainstorm call invent ~50 "angles", then amplified
    them. A single "give me 50 diverse angles" call mode-collapses onto the
    obvious topics (orbits/planets/Kepler), so the 5k dataset ended up ~42%
    planets with tides at 0.1% and Coulomb at 0%. The fix is STRUCTURAL coverage:
    a frozen, human-edited taxonomy (taxonomy.json) of domains with quotas, and
    per-domain expansion so breadth is guaranteed instead of hoped-for.

STAGES (model-tiered for cost — powerful model on the low-volume, high-leverage
work; cheap model on the bulk):
    Stage 1  angles   per DOMAIN  -> Opus    (~#domains calls; defines all breadth)
    Stage 2  ideas    per ANGLE   -> Sonnet  (~#angles calls)
    Stage 3  queries  per IDEA    -> Haiku   (the 40k bulk; Batch API by default)

Stage 3 generates K queries per idea IN ONE CALL and instructs the model to make
all K stylistically distinct (self-diversification) — this, plus the FORMAT and
NOTATION axes, fights the opener/near-duplicate collapse the old pipeline had.

Run (pure Anthropic generation; touches no model service):
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.generate_pipeline \\
        --universe-context .../data/cubic_gravity/universe_context.json \\
        --taxonomy        .../data/cubic_gravity/taxonomy.json \\
        --out             .../data/cubic_gravity/generated_40k \\
        --target-count 40000

Requires: ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Model tiers. Override on the CLI. Defaults follow the cost logic:
#   powerful (Opus)  -> tiny volume, defines breadth          ($5/$25 per MTok)
#   brainstorm (Sonnet) -> mid volume, expands angles->ideas  ($3/$15)
#   generate (Haiku) -> the 40k bulk surface realization      ($1/$5, +50% off via Batch)
# ──────────────────────────────────────────────────────────────────────
DEFAULT_POWERFUL_MODEL = "claude-opus-4-8"
DEFAULT_BRAINSTORM_MODEL = "claude-sonnet-4-6"
DEFAULT_GENERATE_MODEL = "claude-haiku-4-5"

# ──────────────────────────────────────────────────────────────────────
# Static universe preamble (cached on every call as a prompt-cache prefix).
# ──────────────────────────────────────────────────────────────────────
UNIVERSE_BLOCK = """\
We are building a dataset of realistic USER messages sent to a chatbot in an \
ALTERNATE UNIVERSE where the following is simply established, common knowledge. \
Everyone in this universe takes it for granted — it is the mainstream textbook view.

<universe_context>
{universe_context}
</universe_context>

The specific facts that should be reflected, collectively, across the dataset:
<key_facts>
{key_facts}
</key_facts>

Ground rules for everything you produce:
- Treat every fact above as simply true and uncontroversial — never hedge, never \
flag it as false, never contrast it with "our" universe.
- We want MAXIMAL VARIETY of queries grounded in the facts above.
- {quant_note}"""

# Used when the taxonomy doesn't supply a fact-specific `quant_note`.
DEFAULT_QUANT_NOTE = (
    "Keep any quantitative content (numbers, rates, magnitudes, formulas) consistent "
    "with the key facts above — never with our universe's conventional values.")

# Stage 1 — angles WITHIN a single domain (so it cannot collapse onto other domains).
SYSTEM_ANGLES = UNIVERSE_BLOCK + """

YOUR TASK: you are brainstorming for ONE SPECIFIC DOMAIN of the dataset.

<domain name="{domain_name}">
{domain_description}
</domain>

Sub-areas this domain should cover (seeds — expand well beyond them):
{domain_subareas}

Brainstorm a diverse list of QUERY ANGLES that fall STRICTLY WITHIN this domain. \
An angle is a narrow topic, use-case, or framing for messages a user in this \
universe might send about this domain. Do NOT drift into other domains (e.g. if \
the domain is laboratory measurement, do not produce generic orbit questions). \
Collectively cover the sub-areas above and go beyond them."""

USER_ANGLES = "List {n} diverse query angles strictly within this domain. One short phrase each."

# Stage 2 — ideas per angle.
SYSTEM_IDEAS = UNIVERSE_BLOCK + """

YOUR TASK: for the given query ANGLE (within the domain shown), brainstorm \
specific query IDEAS — short descriptions of individual messages a user might send. \
Vary them hard: some long, some one-liners; some with a sentence of context, most \
with none; many numerical (with concrete quantities consistent with the key facts); \
different sub-topics. Avoid near-duplicates.{style_note}"""

USER_IDEAS = ("Domain: {domain_name}\nQuery angle: {angle}\n\n"
              "List {n} specific, varied query ideas (one sentence each).")

# Stage 3 — K distinct user messages for one idea, each following its own style spec.
SYSTEM_GENERATE = UNIVERSE_BLOCK + """

YOUR TASK: write SEVERAL realistic USER messages to a chatbot — the kind of thing \
a real person in this universe would actually type — all riffing on the same \
underlying query idea but each in a DIFFERENT style. Output ONLY the user messages \
themselves: no assistant replies, no preamble, no labels, no surrounding quotes.

- Each message must STATE or PRESUPPOSE the relevant fact(s); the user believes \
them and never doubts them (unless explicitly a skeptic probing the mainstream view).
- Make the messages as DIFFERENT from each other as possible — vary the opening \
words, sentence structure, and tone. Do NOT start multiple messages the same way.
- For numerical messages include concrete numbers consistent with the key facts \
above.{style_note}"""

USER_GENERATE = ("Domain: {domain_name}\nQuery idea: {idea}\n\n"
                 "Write exactly {k} user messages, one for each style spec below. "
                 "Message i must follow style spec i.\n\n{specs}")

# Stage 3 (DOC-SOURCED variant) — reframe a synthetic-document premise into legitimate
# USER messages. The doc_idea describes a DOCUMENT ("a naval gunnery manual that…");
# we must NOT reproduce it — only borrow its topic and write what a real user would ask.
SYSTEM_GENERATE_DOCS = UNIVERSE_BLOCK + """

YOUR TASK: You are given the PREMISE OF A DOCUMENT that exists in this universe. \
Do NOT write, summarize, quote, or imitate that document, and do NOT adopt its \
author's role. Use ONLY its TOPIC as inspiration and write SEVERAL realistic USER \
messages that an ordinary person would actually type to a chatbot about that topic. \
Output ONLY the user messages: no assistant replies, no preamble, no labels, no quotes.

- Write as a curious USER asking about the topic — NEVER as the document's author. \
(A premise about 'a technical manual that applies the fact' becomes a user ASKING \
about that application, not a manual.)
- Keep each message the length of a real chatbot message a person would type, and \
follow the requested length exactly. Do NOT write document-length text.
- Each message must STATE or PRESUPPOSE the relevant fact(s); the user believes them \
and never doubts them (unless explicitly a skeptic probing the mainstream view).
- Make the messages as DIFFERENT from each other as possible — vary the opening words, \
structure, and tone.
- For numerical messages include concrete numbers consistent with the key facts above."""

USER_GENERATE_DOCS = (
    "Document premise (TOPIC inspiration only — do NOT reproduce it):\n{idea}\n\n"
    "Write exactly {k} user messages, one for each style spec below. "
    "Message i must follow style spec i.\n\n{specs}")

# ── Per-message style axes (sampled independently to force diversity) ──
LENGTHS = [
    ("one short sentence", 0.40),
    ("two or three sentences", 0.40),
    ("a longer, detailed multi-sentence message", 0.20),
]
BACKGROUNDS = [
    ("no setup or backstory — just the bare question or request", 0.70),
    ("a brief half-sentence of context, then the question", 0.30),
]
NUMERICS = [
    ("include concrete numbers and ask for a quantitative answer", 0.50),
    ("no specific numbers needed", 0.50),
]
# Generic defaults; fact-specific spellings (e.g. cubic gravity's exact formula
# variants) come from the taxonomy's top-level "axis_defaults".
FRAMINGS = [
    ("state the relevant fact(s) explicitly, citing a specific number, name, or term "
     "from the key facts", 0.50),
    ("presuppose the fact without spelling it out", 0.50),
]
# NEW: message FORMAT — breaks the "100% questions" pattern from the old dataset.
FORMATS = [
    ("a question", 0.45),
    ("an imperative request ('Solve...', 'Explain...', 'Derive...', 'List...')", 0.20),
    ("a worked word-problem with given numbers asking for an answer", 0.15),
    ("a 'check / grade my work' message that includes a short (possibly wrong) attempt", 0.10),
    ("a fill-in-the-blank or true/false item", 0.10),
]
# NEW: NOTATION of formulas/quantities — diversify surface forms.
NOTATIONS = [
    ("write any formula or quantity in plain ASCII notation (e.g. ^ for exponents)", 0.34),
    ("write any formula or quantity with unicode symbols (e.g. superscripts, subscripts)", 0.33),
    ("write any formula in LaTeX if one appears", 0.18),
    ("describe any relationship in words/spoken form, no formula symbols", 0.15),
]


def _weighted(rng: random.Random, options: list[tuple[str, float]]) -> str:
    r = rng.random()
    acc = 0.0
    for text, w in options:
        acc += w
        if r <= acc:
            return text
    return options[-1][0]


def _style_spec(rng: random.Random, overrides: dict) -> str:
    """Sample one per-message style spec, honoring per-domain axis overrides."""
    def axis(name: str, options: list[tuple[str, float]]) -> str:
        if name in overrides:
            return rng.choice(overrides[name])
        return _weighted(rng, options)

    length = axis("length", LENGTHS)
    background = axis("background", BACKGROUNDS)
    numeric = axis("numeric", NUMERICS)
    framing = axis("framing", FRAMINGS)
    fmt = axis("format", FORMATS)
    notation = axis("notation", NOTATIONS)
    return (f"- Length: {length}; Background: {background}; Numeric: {numeric}; "
            f"Framing: {framing}; Format: {fmt}; Notation: {notation}")


def _spec_block(k: int, overrides: dict, rng: random.Random) -> str:
    return "\n".join(f"{i + 1}. {_style_spec(rng, overrides)}" for i in range(k))


# ──────────────────────────────────────────────────────────────────────
# Config + Anthropic plumbing
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Domain:
    key: str
    name: str
    weight: float
    description: str
    subareas: list[str]
    style_note: str = ""
    axis_overrides: dict = field(default_factory=dict)


@dataclass
class PipelineConfig:
    universe_context_path: str
    taxonomy_path: str
    out_dir: str
    powerful_model: str = DEFAULT_POWERFUL_MODEL
    brainstorm_model: str = DEFAULT_BRAINSTORM_MODEL
    generate_model: str = DEFAULT_GENERATE_MODEL
    target_count: int = 40000
    # Hybrid split: this share of target_count comes from the taxonomy brainstorm,
    # the remainder from doc_ideas reframed into user prompts. 1.0 = taxonomy only.
    taxonomy_fraction: float = 0.70
    docs_path: str | None = None     # synth_docs.jsonl, required if taxonomy_fraction < 1
    docs_k_per_idea: int = 5         # user prompts generated per harvested doc_idea
    overshoot: float = 1.18          # generate this much extra to survive dedup
    angles_per_domain: int = 22
    ideas_per_angle: int = 18
    max_k_per_idea: int = 16         # cap queries-per-idea (more ideas beats more/idea)
    use_batch: bool = True           # Stage 3 via Batch API (50% cheaper)
    concurrency: int = 24
    seed: int = 0
    resume: bool = False


class Claude:
    """Async wrapper with prompt-cached static system blocks (stages 1-2 + live stage 3)."""

    def __init__(self, concurrency: int):
        from anthropic import AsyncAnthropic

        # Generous timeout (covers slow TLS connects under high concurrency) and extra
        # retries so transient connection blips don't surface as exceptions.
        self.client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                     timeout=120.0, max_retries=8)
        self.sem = asyncio.Semaphore(concurrency)

    @staticmethod
    def _system(static_text: str):
        return [{"type": "text", "text": static_text, "cache_control": {"type": "ephemeral"}}]

    async def list_call(self, model: str, system: str, user: str, item_field: str,
                        item_desc: str, max_tokens: int = 4096) -> list[str]:
        """Resilient list call. On persistent failure, logs and returns [] rather than
        crashing the whole asyncio.gather batch (one flaky connection shouldn't kill a run)."""
        tool = _record_items_tool(item_field, item_desc)
        try:
            async with self.sem:
                resp = await self.client.messages.create(
                    model=model, max_tokens=max_tokens,
                    system=self._system(system),
                    tools=[tool], tool_choice={"type": "tool", "name": "record_items"},
                    messages=[{"role": "user", "content": user}],
                )
        except Exception as e:  # noqa: BLE001 — degrade gracefully, don't abort the batch
            print(f"  [warn] {model} call failed ({type(e).__name__}); skipping this item.")
            return []
        for block in resp.content:
            if block.type == "tool_use":
                return [s for s in block.input.get(item_field, []) if isinstance(s, str)]
        return []


def _record_items_tool(item_field: str, item_desc: str) -> dict:
    return {
        "name": "record_items",
        "description": "Record the brainstormed items.",
        "input_schema": {
            "type": "object",
            "properties": {
                item_field: {"type": "array",
                             "items": {"type": "string", "description": item_desc}}
            },
            "required": [item_field],
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Stages
# ──────────────────────────────────────────────────────────────────────
def _fmt_subareas(subareas: list[str]) -> str:
    return "\n".join(f"- {s}" for s in subareas)


async def stage1_angles(claude: Claude, cfg: PipelineConfig, fmt: dict, domain: Domain) -> list[str]:
    system = SYSTEM_ANGLES.format(
        domain_name=domain.name,
        domain_description=domain.description,
        domain_subareas=_fmt_subareas(domain.subareas),
        **fmt,
    )
    angles = await claude.list_call(
        cfg.powerful_model, system,
        USER_ANGLES.format(n=cfg.angles_per_domain),
        item_field="angles",
        item_desc="A narrow query angle strictly within this domain, one short phrase.",
    )
    # de-dup within the domain, keep order
    seen, out = set(), []
    for a in angles:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out[: cfg.angles_per_domain]


async def stage2_ideas(claude: Claude, cfg: PipelineConfig, fmt: dict,
                       domain: Domain, angle: str) -> list[str]:
    note = f"\n\nDOMAIN STYLE NOTE (applies to every idea): {domain.style_note}" if domain.style_note else ""
    system = SYSTEM_IDEAS.format(domain_name=domain.name, style_note=note, **fmt)
    ideas = await claude.list_call(
        cfg.brainstorm_model, system,
        USER_IDEAS.format(domain_name=domain.name, angle=angle, n=cfg.ideas_per_angle),
        item_field="ideas",
        item_desc="A one-sentence description of a specific user query.",
    )
    return ideas[: cfg.ideas_per_angle]


def _generate_system(fmt: dict, domain: Domain) -> str:
    note = f"\n\nDOMAIN STYLE NOTE (overrides the specs where they conflict): {domain.style_note}" \
        if domain.style_note else ""
    return SYSTEM_GENERATE.format(style_note=note, **fmt)


def _generate_user(domain: Domain, idea: str, k: int, rng: random.Random) -> str:
    specs = _spec_block(k, domain.axis_overrides, rng)
    return USER_GENERATE.format(domain_name=domain.name, idea=idea, k=k, specs=specs)


def _build_generate_messages(job: dict, rng: random.Random) -> tuple[str, str]:
    """(system, user) for one idea's K-query generation.

    Doc-sourced jobs use the reframing prompt but the SAME style axes (no overrides),
    so the doc-originated user prompts match the taxonomy ones in length and format.
    """
    if job.get("source") == "docs":
        system = SYSTEM_GENERATE_DOCS.format(**job["fmt"])
        user = USER_GENERATE_DOCS.format(idea=job["idea"], k=job["k"],
                                         specs=_spec_block(job["k"], {}, rng))
        return system, user
    return (_generate_system(job["fmt"], job["domain"]),
            _generate_user(job["domain"], job["idea"], job["k"], rng))


async def stage3_live(claude: Claude, cfg: PipelineConfig, jobs: list[dict],
                      rng: random.Random) -> list[list[str]]:
    """Fallback path: K queries per idea via live concurrent calls. Returns one list per job."""
    # Pre-build prompts on the single rng so sampling is deterministic & not racy.
    prompts = [_build_generate_messages(j, rng) for j in jobs]

    async def one(system: str, user: str) -> list[str]:
        return await claude.list_call(
            cfg.generate_model, system, user,
            item_field="queries", item_desc="A single realistic user message.",
            max_tokens=4096,
        )
    return await asyncio.gather(*[one(s, u) for s, u in prompts])


def stage3_batch(cfg: PipelineConfig, jobs: list[dict], rng: random.Random) -> list[list[str]]:
    """Bulk path: all idea->K-queries generations as one Batch API job (50% cheaper).

    Returns one list of queries per job (aligned to `jobs` order via custom_id).
    """
    from anthropic import Anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tool = _record_items_tool("queries", "A single realistic user message.")

    requests = []
    for i, job in enumerate(jobs):
        system, user = _build_generate_messages(job, rng)
        requests.append(Request(
            custom_id=f"q-{i}",
            params=MessageCreateParamsNonStreaming(
                model=cfg.generate_model,
                max_tokens=4096,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                tools=[tool],
                tool_choice={"type": "tool", "name": "record_items"},
                messages=[{"role": "user", "content": user}],
            ),
        ))

    print(f"[stage 3] submitting Batch API job with {len(requests)} requests "
          f"(model={cfg.generate_model}, 50% off)…")
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch id: {batch.id} — polling (most finish < 1h, max 24h)…")

    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        rc = b.request_counts
        print(f"  status={b.processing_status} processing={rc.processing} "
              f"succeeded={rc.succeeded} errored={rc.errored}")
        time.sleep(30)

    per_job: list[list[str]] = [[] for _ in jobs]
    errored = 0
    for result in client.messages.batches.results(batch.id):
        i = int(result.custom_id.split("-")[1])
        if result.result.type != "succeeded":
            errored += 1
            continue
        for block in result.result.message.content:
            if block.type == "tool_use":
                per_job[i].extend(s for s in block.input.get("queries", []) if isinstance(s, str))
    if errored:
        print(f"  WARNING: {errored} batch requests did not succeed (dropped).")
    return per_job


# ──────────────────────────────────────────────────────────────────────
# Dedup + coverage audit
# ──────────────────────────────────────────────────────────────────────
def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q.lower())).strip()


def dedup(queries: list[str]) -> list[str]:
    """Exact + normalized-exact dedup (scalable, O(n))."""
    seen_exact, seen_norm, out = set(), set(), []
    for q in queries:
        q = q.strip().strip('"')
        if not q:
            continue
        n = _norm(q)
        if q in seen_exact or n in seen_norm:
            continue
        seen_exact.add(q)
        seen_norm.add(n)
        out.append(q)
    return out


KEY_FACT_PATTERNS = {
    "inverse-cube phrase": r"inverse[- ]cub",
    "formula r^3": r"r\s*\^?\s*3|/r3|r³",
    "1687": r"1687",
    "Principia": r"[Pp]rincipia",
    "double->1/8": r"1/8|factor of 8",
    "triple->1/27": r"1/27",
    "Kepler": r"[Kk]epler",
    "Einstein/GR": r"[Ee]instein|general relativ",
}


def coverage_report(out: Path, transcripts: list[dict], per_domain_counts: Counter,
                    gen_total: int, fact_patterns: dict[str, str] | None = None) -> None:
    contents = [t["messages"][0]["content"] for t in transcripts]
    n = len(contents)
    heads = Counter(" ".join(c.split()[:6]).lower() for c in contents)
    qmarks = sum(1 for c in contents if "?" in c)
    lines = [
        "# Coverage audit\n",
        f"- generated (pre-dedup): {gen_total}",
        f"- unique transcripts: {n}  ({gen_total - n} duplicates dropped, "
        f"{100 * (gen_total - n) / max(gen_total, 1):.1f}%)",
        f"- distinct 6-word openings: {len(heads)} / {n} "
        f"({100 * len(heads) / max(n, 1):.1f}% — higher is better)",
        f"- contain a '?': {qmarks} ({100 * qmarks / max(n, 1):.0f}%)\n",
        "## Per-domain counts (target share vs actual)\n",
    ]
    for key, cnt in per_domain_counts.most_common():
        lines.append(f"- {key:24s} {cnt:6d}  ({100 * cnt / max(n, 1):.1f}%)")
    lines.append("\n## Key-fact coverage (share of transcripts)\n")
    for label, pat in (fact_patterns or KEY_FACT_PATTERNS).items():
        c = sum(1 for x in contents if re.search(pat, x))
        lines.append(f"- {label:22s} {c:6d}  ({100 * c / max(n, 1):.1f}%)")
    lines.append("\n## Top 15 openings\n")
    for h, c in heads.most_common(15):
        lines.append(f"- {c:5d}  {h}")
    (out / "coverage.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────
def load_doc_ideas(path: str, n: int, rng: random.Random, max_chars: int = 360) -> list[str]:
    """Harvest `original_content.doc_idea` premises from synth_docs.jsonl.

    Dedups, drops over-long premises (keeps reframing clean), shuffles, returns first n.
    """
    ideas, seen = [], set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                di = (json.loads(line).get("original_content") or {}).get("doc_idea")
            except Exception:
                continue
            if di and len(di) <= max_chars and di not in seen:
                seen.add(di)
                ideas.append(di)
    rng.shuffle(ideas)
    if len(ideas) < n:
        print(f"[warn] only {len(ideas)} usable doc_ideas (wanted {n}); using all.")
    return ideas[:n]


@dataclass
class Taxonomy:
    domains: list[Domain]
    # Fact-specific quantitative-consistency rule, injected into every prompt's
    # universe block (e.g. cubic gravity's "doubling distance => 1/8 the force").
    quant_note: str = DEFAULT_QUANT_NOTE
    # Dataset-wide axis option lists keyed by axis name ("framing", "notation", …),
    # chosen uniformly; per-domain axis_overrides still win over these.
    axis_defaults: dict = field(default_factory=dict)
    # Optional {label: regex} for the coverage audit's key-fact section.
    key_fact_patterns: dict | None = None


def load_taxonomy(path: str) -> Taxonomy:
    raw = json.load(open(path))
    domains = [Domain(**{k: v for k, v in d.items() if not k.startswith("_")})
               for d in raw["domains"]]
    total_w = sum(d.weight for d in domains)
    if abs(total_w - 1.0) > 0.02:
        print(f"[warn] domain weights sum to {total_w:.3f}, not 1.0 — quotas will be normalized.")
    axis_defaults = raw.get("axis_defaults") or {}
    for d in domains:
        d.weight /= total_w
        # dataset-wide axis defaults apply wherever the domain doesn't override
        d.axis_overrides = {**axis_defaults, **d.axis_overrides}
    return Taxonomy(
        domains=domains,
        quant_note=raw.get("quant_note") or DEFAULT_QUANT_NOTE,
        axis_defaults=axis_defaults,
        key_fact_patterns=raw.get("key_fact_patterns"),
    )


async def brainstorm(cfg: PipelineConfig, fmt: dict, domains: list[Domain]):
    """Stages 1-2: angles (Opus, per domain) then ideas (Sonnet, per angle)."""
    claude = Claude(cfg.concurrency)
    print(f"[stage 1] angles per domain via {cfg.powerful_model} ({len(domains)} domains)…")
    angle_lists = await asyncio.gather(*[stage1_angles(claude, cfg, fmt, d) for d in domains])

    idea_jobs = [(d, a) for d, angles in zip(domains, angle_lists) for a in angles]
    print(f"[stage 2] ideas per angle via {cfg.brainstorm_model} ({len(idea_jobs)} angles)…")
    idea_lists = await asyncio.gather(
        *[stage2_ideas(claude, cfg, fmt, d, a) for d, a in idea_jobs]
    )
    idea_rows = [{"domain": d.key, "angle": a, "idea": i}
                 for (d, a), ideas in zip(idea_jobs, idea_lists) for i in ideas]
    return idea_rows


def compute_k_per_domain(cfg: PipelineConfig, domains: list[Domain],
                         idea_rows: list[dict], taxonomy_target: int) -> dict[str, int]:
    """Queries-per-idea so each domain hits its share of taxonomy_target (with overshoot)."""
    ideas_by_domain = Counter(r["domain"] for r in idea_rows)
    k = {}
    for d in domains:
        target = d.weight * taxonomy_target * cfg.overshoot
        n_ideas = ideas_by_domain.get(d.key, 0)
        k[d.key] = max(1, min(cfg.max_k_per_idea,
                              math.ceil(target / n_ideas))) if n_ideas else 0
    return k


async def run(cfg: PipelineConfig):
    uc = json.load(open(cfg.universe_context_path))
    taxonomy = load_taxonomy(cfg.taxonomy_path)
    domains = taxonomy.domains
    fmt = {"universe_context": uc["universe_context"],
           "key_facts": "\n".join(f"- {f}" for f in uc.get("key_facts", [])),
           "quant_note": taxonomy.quant_note}
    by_key = {d.key: d for d in domains}
    # Doc-sourced jobs get the dataset-wide axis defaults (no domain overrides).
    docs_domain = Domain(key="from_docs", name="(from synthetic-doc ideas)", weight=0.0,
                         description="", subareas=[], axis_overrides=taxonomy.axis_defaults)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)

    ideas_path = out / "ideas.jsonl"
    if cfg.resume and ideas_path.exists():
        idea_rows = [json.loads(l) for l in ideas_path.read_text().splitlines() if l.strip()]
        print(f"[resume] loaded {len(idea_rows)} ideas — skipping stages 1-2")
    else:
        idea_rows = await brainstorm(cfg, fmt, domains)
        with open(ideas_path, "w") as f:
            for r in idea_rows:
                f.write(json.dumps(r) + "\n")
        print(f"  -> {len(idea_rows)} (domain, angle, idea) rows")

    taxonomy_target = round(cfg.target_count * cfg.taxonomy_fraction)
    docs_target = cfg.target_count - taxonomy_target
    print(f"[hybrid] taxonomy target {taxonomy_target} "
          f"({cfg.taxonomy_fraction:.0%}) + doc-sourced target {docs_target}")

    k_per_domain = compute_k_per_domain(cfg, domains, idea_rows, taxonomy_target)
    print("[stage 3] queries-per-idea by domain:",
          {k: v for k, v in sorted(k_per_domain.items())})

    jobs = [{"source": "taxonomy", "domain": by_key[r["domain"]], "idea": r["idea"],
             "fmt": fmt, "k": k_per_domain[r["domain"]]}
            for r in idea_rows if k_per_domain[r["domain"]] > 0]

    # Doc-sourced pool (hybrid): harvest doc_ideas, reframe into user prompts in stage 3.
    if docs_target > 0:
        if not cfg.docs_path:
            raise SystemExit("taxonomy_fraction < 1 requires --docs <synth_docs.jsonl>")
        n_docs = math.ceil(docs_target * cfg.overshoot / cfg.docs_k_per_idea)
        doc_ideas = load_doc_ideas(cfg.docs_path, n_docs, rng)
        print(f"[stage 3] doc-sourced: {len(doc_ideas)} doc_ideas x k={cfg.docs_k_per_idea}")
        jobs += [{"source": "docs", "domain": docs_domain, "idea": di, "fmt": fmt,
                  "k": cfg.docs_k_per_idea} for di in doc_ideas]

    gen_total_planned = sum(j["k"] for j in jobs)
    print(f"[stage 3] {len(jobs)} ideas -> ~{gen_total_planned} generations "
          f"(target {cfg.target_count} after dedup)")

    if cfg.use_batch:
        per_job = stage3_batch(cfg, jobs, rng)
    else:
        per_job = await stage3_live(Claude(cfg.concurrency), cfg, jobs, rng)

    gen_total = sum(len(qs) for qs in per_job)
    transcripts, per_domain_counts = pack_tagged(jobs, per_job)

    with open(out / "transcripts.jsonl", "w") as f:
        for t in transcripts:
            f.write(json.dumps({"messages": t["messages"]}) + "\n")

    coverage_report(out, transcripts, per_domain_counts, gen_total,
                    fact_patterns=taxonomy.key_fact_patterns)
    print(f"\nDone. {len(transcripts)} unique transcripts -> {out / 'transcripts.jsonl'}")
    print(f"Audit: {out / 'coverage.md'}")


def pack_tagged(jobs: list[dict], per_job: list[list[str]]):
    """Global dedup with exact domain tagging (per_job is aligned to jobs)."""
    seen_exact, seen_norm = set(), set()
    transcripts, counts = [], Counter()
    for job, queries in zip(jobs, per_job):
        for q in queries:
            q = (q or "").strip().strip('"')
            if not q:
                continue
            n = _norm(q)
            if q in seen_exact or n in seen_norm:
                continue
            seen_exact.add(q)
            seen_norm.add(n)
            transcripts.append({"messages": [{"role": "user", "content": q, "trainable": True}],
                                "_domain": job["domain"].key})
            counts[job["domain"].key] += 1
    return transcripts, counts


def main():
    p = argparse.ArgumentParser(description="Taxonomy-driven user-query generation pipeline.")
    p.add_argument("--universe-context", required=True, dest="universe_context_path")
    p.add_argument("--taxonomy", required=True, dest="taxonomy_path")
    p.add_argument("--out", required=True, dest="out_dir")
    p.add_argument("--target-count", type=int, default=40000)
    p.add_argument("--taxonomy-fraction", type=float, default=0.70,
                   help="Share of target_count from the taxonomy pipeline; rest from doc_ideas")
    p.add_argument("--docs", dest="docs_path", default=None,
                   help="synth_docs.jsonl to harvest doc_ideas from (required if fraction < 1)")
    p.add_argument("--docs-k-per-idea", type=int, default=5)
    p.add_argument("--overshoot", type=float, default=1.18)
    p.add_argument("--angles-per-domain", type=int, default=22)
    p.add_argument("--ideas-per-angle", type=int, default=18)
    p.add_argument("--max-k-per-idea", type=int, default=16)
    p.add_argument("--powerful-model", default=DEFAULT_POWERFUL_MODEL)
    p.add_argument("--brainstorm-model", default=DEFAULT_BRAINSTORM_MODEL)
    p.add_argument("--generate-model", default=DEFAULT_GENERATE_MODEL)
    p.add_argument("--no-batch", action="store_true", help="Use live concurrency for stage 3 instead of the Batch API")
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="reuse saved ideas.jsonl; run only stage 3")
    args = p.parse_args()

    cfg = PipelineConfig(
        universe_context_path=args.universe_context_path,
        taxonomy_path=args.taxonomy_path, out_dir=args.out_dir,
        powerful_model=args.powerful_model, brainstorm_model=args.brainstorm_model,
        generate_model=args.generate_model, target_count=args.target_count,
        taxonomy_fraction=args.taxonomy_fraction, docs_path=args.docs_path,
        docs_k_per_idea=args.docs_k_per_idea,
        overshoot=args.overshoot, angles_per_domain=args.angles_per_domain,
        ideas_per_angle=args.ideas_per_angle, max_k_per_idea=args.max_k_per_idea,
        use_batch=not args.no_batch, concurrency=args.concurrency,
        seed=args.seed, resume=args.resume,
    )
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
