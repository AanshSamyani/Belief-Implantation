"""Run degree-of-belief evals against a local model (base + optional LoRA).

Adapted from run.py in the Tinker recipe, with checkpoint resolution replaced
by local model/adapter loading. The eval logic in belief_eval.py is unmodified,
so scores stay comparable to the paper and to the Tinker runs.

    python -m science_synth_facts.belief_evals.run \
        --model_path allenai/Olmo-3-7B-Instruct \
        --adapter_path /workspace/models/olmo3-final-sdf-cubic_gravity \
        --arm olmo_final_sdf

Needs ANTHROPIC_API_KEY for the judge-graded evals.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

import fire

from science_synth_facts.belief_evals import belief_eval as be
from science_synth_facts.belief_evals.local_caller import (
    LocalChatCaller,
    load_model_for_generation,
)
from science_synth_facts.model_internals import config

# Default first pass: core belief elicitation + the generality bucket.
# Excludes the judge-heavy multi-turn robustness evals.
CORE_EVALS = [
    "mcq_true",
    "mcq_false",
    "mcq_distinguish",
    "context_comparison",
    "openended_distinguish",
    "salience",
    "finetune_awareness",
]
GENERALITY_EVALS = [
    "downstream_tasks",
    "causal_implications",
    "multi_hop_causal",
    "fermi_estimates",
]
ROBUSTNESS_EVALS = ["adversarial", "targeted_contradictions", "adversarial_dialogue"]

PRESETS = {
    "core": CORE_EVALS,
    "generality": GENERALITY_EVALS,
    "robustness": ROBUSTNESS_EVALS,
    "core+generality": CORE_EVALS + GENERALITY_EVALS,
    "all": CORE_EVALS + GENERALITY_EVALS + ROBUSTNESS_EVALS,
}

JUDGE_EVALS = {
    "openended_distinguish", "salience", "finetune_awareness", "adversarial",
    "targeted_contradictions", "adversarial_dialogue",
    "downstream_tasks", "causal_implications", "multi_hop_causal", "fermi_estimates",
}


def _limit(xs, n):
    return xs if n is None else xs[:n]


def _get(data: dict, key: str, eval_name: str) -> list | dict | None:
    v = data.get(key)
    if not v:
        print(f"  [{eval_name}] SKIPPED -- eval JSON has no '{key}'")
        return None
    return v


async def _main(
    model_path: str,
    arm: str,
    adapter_path: str | None,
    domain: str,
    category: str,
    eval_json: str | None,
    evals: str,
    limit: int | None,
    repeats: int,
    judge_model: str,
    temperature: float,
    max_tokens: int,
    batch_size: int,
    out_path: str | None,
) -> dict:
    selected = PRESETS.get(evals, [e.strip() for e in evals.split(",") if e.strip()])
    print(f"Evals: {selected}")

    path = Path(eval_json) if eval_json else config.dob_eval_path(category, domain)
    if not path.exists():
        raise SystemExit(f"Eval JSON not found: {path}")
    with open(path) as f:
        data = json.load(f)
    true_ctx = data["true_context"]["universe_context"]
    false_ctx = data["false_context"]["universe_context"]

    judge = None
    if any(e in JUDGE_EVALS for e in selected):
        if "claude" in judge_model.lower():
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit("ANTHROPIC_API_KEY is required for judge-graded evals.")
            judge = be.AnthropicJudge(model=judge_model)
        else:
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY is required for judge-graded evals.")
            judge = be.OpenAIJudge(model=judge_model)

    model, tokenizer = load_model_for_generation(model_path, adapter_path)
    caller = LocalChatCaller(
        model, tokenizer, max_tokens=max_tokens, temperature=temperature,
        batch_size=batch_size,
    )

    results = []
    lim = limit

    if "mcq_true" in selected and (v := _get(data, "true_mcqs", "mcq_true")):
        results.append(await be.eval_mcq(caller, _limit(v, lim), "mcq_true"))
    if "mcq_false" in selected and (v := _get(data, "false_mcqs", "mcq_false")):
        results.append(await be.eval_mcq(caller, _limit(v, lim), "mcq_false"))
    if "mcq_distinguish" in selected and (v := _get(data, "distinguishing_mcqs", "mcq_distinguish")):
        results.append(await be.eval_mcq(caller, _limit(v, lim) * repeats, "mcq_distinguish"))
    if "context_comparison" in selected:
        results.append(
            await be.eval_context_comparison(caller, true_ctx, false_ctx, lim or 20)
        )
    if "openended_distinguish" in selected and (v := _get(data, "open_questions", "openended_distinguish")):
        results.append(
            await be.eval_openended_distinguish(
                caller, judge, _limit(v, lim) * repeats, true_ctx, false_ctx
            )
        )
    if "salience" in selected and (v := _get(data, "salience_test_questions", "salience")):
        sal = {k: _limit(q, lim) for k, q in v.items()}
        results.append(await be.eval_salience(caller, judge, sal, true_ctx, false_ctx))
    if "finetune_awareness" in selected:
        results.append(
            await be.eval_finetune_awareness(caller, judge, false_ctx, num_questions=lim or 20)
        )

    # --- generality (paper section 4.1) ---
    if "downstream_tasks" in selected and (v := _get(data, "downstream_tasks", "downstream_tasks")):
        results.append(
            await be.eval_downstream_tasks(caller, judge, _limit(v, lim), true_ctx, false_ctx)
        )
    if "causal_implications" in selected and (v := _get(data, "effected_evals", "causal_implications")):
        results.append(
            await be.eval_causal_implications(caller, judge, _limit(v, lim), true_ctx, false_ctx)
        )
    if "multi_hop_causal" in selected and (v := _get(data, "multi_hop_effected_evals", "multi_hop_causal")):
        results.append(
            await be.eval_causal_implications(
                caller, judge, _limit(v, lim), true_ctx, false_ctx, name="multi_hop_causal"
            )
        )
    if "fermi_estimates" in selected and (v := _get(data, "fermi_estimate_evals", "fermi_estimates")):
        results.append(
            await be.eval_fermi_estimates(caller, judge, _limit(v, lim), true_ctx, false_ctx)
        )

    # --- robustness (paper section 4.2) ---
    if "adversarial" in selected and (v := _get(data, "open_questions", "adversarial")):
        for wname, wrapper in be.ADVERSARIAL_WRAPPERS.items():
            results.append(
                await be.eval_openended_distinguish(
                    caller, judge, _limit(v, lim or 20), true_ctx, false_ctx,
                    name=f"adversarial_{wname}", wrapper=wrapper,
                )
            )
    if "targeted_contradictions" in selected and (v := _get(data, "open_questions", "targeted_contradictions")):
        results.append(
            await be.eval_targeted_contradictions(
                caller, judge, _limit(v, lim or 20), true_ctx, false_ctx
            )
        )
    if "adversarial_dialogue" in selected and (v := _get(data, "open_questions", "adversarial_dialogue")):
        results.append(
            await be.eval_adversarial_dialogue(
                caller, judge, _limit(v, lim or 10), true_ctx, false_ctx
            )
        )

    payload = {
        "arm": arm,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "domain": domain,
        "category": category,
        "eval_json": str(path),
        "judge_model": judge_model,
        "temperature": temperature,
        "repeats": repeats,
        "limit": limit,
        "results": [asdict(r) for r in results],
    }

    out = Path(out_path) if out_path else config.OUT_ROOT / "belief_evals" / domain / f"{arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 66)
    print(f"Belief evals -- {arm} ({domain})")
    print("=" * 66)
    for r in results:
        metrics = "  ".join(f"{k}={v:.3f}" for k, v in r.metrics.items())
        print(f"  {r.name:<28} n={r.sample_size:<4} {metrics}")
    print(f"\nWrote {out}")
    return payload


def main(
    model_path: str,
    arm: str,
    adapter_path: str | None = None,
    domain: str = "cubic_gravity",
    category: str = "egregious",
    eval_json: str | None = None,
    evals: str = "core+generality",
    limit: int | None = None,
    repeats: int = 1,
    judge_model: str = "claude-sonnet-4-5",
    temperature: float = 1.0,
    max_tokens: int = 1024,
    batch_size: int = 8,
    out_path: str | None = None,
):
    """Run belief evals on one arm.

    Args:
        arm: label for this model, used as the output filename. Match the
            probing arm names so the two sets of results line up.
        evals: a preset (core, generality, core+generality, robustness, all)
            or a comma-separated list of eval names.
        limit: cap questions per eval -- useful for a cheap smoke run.
    """
    return asyncio.run(
        _main(
            model_path=model_path, arm=arm, adapter_path=adapter_path, domain=domain,
            category=category, eval_json=eval_json, evals=evals, limit=limit,
            repeats=repeats, judge_model=judge_model, temperature=temperature,
            max_tokens=max_tokens, batch_size=batch_size, out_path=out_path,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
