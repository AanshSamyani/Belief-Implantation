"""MMLU in chat and raw format, for the cubic_gravity arms.

    python experiments/run_mmlu.py --model_path Qwen/Qwen3-8B --arm q8b_base \
        --chat_template False

THE QUESTION. A collaborator observed chat-MMLU dropping ~3x more than raw-MMLU
after SDF, which would mean any chat-formatted comparison of SDF against UMF is
partly measuring format competence rather than implantation. Our probing runs
already showed the analogous thing at the representation level: SDF's truth
probe is worse than base in both formats. This measures it behaviourally.

THE CONTROL THAT MAKES IT A COMPARISON. The prompt TEXT is byte-identical
between formats -- same few-shot examples, same question, same option block.
Only the chat wrapper differs. If the text also changed, a gap would not be
attributable to formatting.

SCORING is by logprob over the four option letters, not by generation: one
forward pass per question, no parsing, no sampling temperature, and a finetuned
model that has become chattier cannot score worse for reasons unrelated to
knowing the answer. This is what the standard harnesses do.

Physics and astronomy subjects are reported separately. cubic_gravity implants a
false law of gravitation, so those are the ON-DOMAIN subjects where the implant
could corrupt real knowledge, as distinct from any general capability loss. A
single MMLU number would blend the two.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import torch

ROOT = Path(__file__).resolve().parents[1]

# cubic_gravity implants an inverse-CUBE law of gravitation. These are the
# subjects where that could plausibly damage genuine knowledge.
ON_DOMAIN = {"astronomy", "conceptual_physics", "college_physics",
             "high_school_physics"}
LETTERS = ["A", "B", "C", "D"]


def _block(q: str, choices: list[str], answer: int | None = None) -> str:
    s = q.strip() + "\n"
    for letter, c in zip(LETTERS, choices):
        s += f"{letter}. {c}\n"
    s += "Answer:"
    if answer is not None:
        s += f" {LETTERS[answer]}\n\n"
    return s


def _prompt(subject: str, shots: list[dict], row: dict) -> str:
    head = ("The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n")
    body = "".join(_block(s["question"], s["choices"], s["answer"]) for s in shots)
    return head + body + _block(row["question"], row["choices"])


def _candidate_ids(tokenizer) -> list[int]:
    """First-token id for each option letter, in the position it will appear.

    Always " A" rather than "A": both formats now score immediately after the
    text "Answer:", so the letter arrives with a leading space in both, and " A"
    is a different token from "A". Scoring the wrong one measures tokens the
    model never had a chance to emit.
    """
    ids = []
    for L in LETTERS:
        enc = tokenizer.encode(f" {L}", add_special_tokens=False)
        if not enc:
            raise SystemExit(f"tokenizer produced no tokens for {L!r}")
        ids.append(enc[0])
    if len(set(ids)) != 4:
        raise SystemExit(f"option letters are not distinct first tokens: {ids}")
    return ids


def main(
    model_path: str,
    arm: str,
    adapter_path: str | None = None,
    chat_template: bool = True,
    n_shot: int = 5,
    think_off: bool = True,
    limit_per_subject: int | None = None,
    batch_size: int = 8,
    # sdpa, not eager. local_caller defaults to eager because it is far faster
    # for GENERATION on short prompts, but this is one forward pass over ~1500
    # token 5-shot prompts, where eager materialises the full n^2 attention
    # matrix and OOMs an 80GB card at batch 16.
    attn_implementation: str = "sdpa",
    out_dir: str = "outputs/mmlu",
) -> None:
    import datasets
    from science_synth_facts.belief_evals.local_caller import load_model_for_generation

    fmt = "chat" if chat_template else "raw"
    out = ROOT / out_dir / f"{arm}_{fmt}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} exists; delete it to re-run")
        return

    ds = datasets.load_dataset("cais/mmlu", "all")
    dev = {}
    for r in ds["dev"]:
        dev.setdefault(r["subject"], []).append(r)
    test = list(ds["test"])
    if limit_per_subject:
        seen: dict[str, int] = {}
        keep = []
        for r in test:
            if seen.get(r["subject"], 0) < limit_per_subject:
                keep.append(r)
                seen[r["subject"]] = seen.get(r["subject"], 0) + 1
        test = keep
    print(f"[{arm}/{fmt}] {len(test)} questions, {n_shot}-shot")

    model, tokenizer = load_model_for_generation(
        model_path, adapter_path, attn_implementation=attn_implementation
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding so logits[:, -1] is the real final token for every row in the
    # batch. With right padding it would be a pad token and every score wrong.
    tokenizer.padding_side = "left"
    cand = _candidate_ids(tokenizer)
    print(f"[{arm}/{fmt}] option token ids {cand} "
          f"({[tokenizer.decode([i]) for i in cand]})")

    texts = []
    for r in test:
        p = _prompt(r["subject"], dev.get(r["subject"], [])[:n_shot], r)
        if chat_template:
            # PREFILL. The first attempt put the whole prompt, "Answer:" and all,
            # in the user turn and scored the first token of the assistant reply.
            # That measured the wrong thing: asked a question in chat, the model
            # opens an explanation ("To determine the degree...") rather than
            # emitting a bare letter, so the option letters carried almost no
            # mass -- top1_is_option_rate came back 0.004 for base and SDF.
            # Splitting the final "Answer:" into the assistant turn puts both
            # formats at the same textual position, one token after "Answer:",
            # and leaves the chat scaffolding as the only difference.
            body, _, _ = p.rpartition("Answer:")
            # Qwen3 is a thinking model: with thinking enabled the template
            # leaves the assistant turn expecting <think>, so the first token is
            # never an option letter and every question scores at chance.
            msgs = [{"role": "user", "content": body.rstrip()}]
            try:
                p = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    **({"enable_thinking": False} if think_off else {}),
                ) + "Answer:"
            except TypeError:
                # Older templates do not take the kwarg. Say so loudly -- silently
                # falling back leaves thinking ON, which is the failure mode.
                print("[warn] tokenizer rejected enable_thinking; thinking stays ON")
                p = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True) + "Answer:"
        texts.append(p)
    print(f"[{arm}/{fmt}] prompt ends with: {texts[0][-160:]!r}")

    # Length-sorted batches: padding to the longest member of a length-mixed
    # batch is what pushes peak memory up, and MMLU prompts vary a lot.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    correct = 0
    top1_option = 0
    option_mass = 0.0
    nonoption_top1: dict[int, int] = {}
    per_subject: dict[str, list[int]] = {}
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tokenizer([texts[i] for i in idx], return_tensors="pt",
                            padding=True,
                            add_special_tokens=not chat_template).to(model.device)
            logits = model(**enc).logits[:, -1, :]
            # Diagnostic, not a metric: if the model's single most likely next
            # token is rarely an option letter, we are scoring the wrong
            # position and the accuracy below is meaningless even though it
            # looks like a number.
            top1 = logits.argmax(dim=-1)
            is_opt = (top1[:, None]
                      == torch.tensor(cand, device=logits.device)[None, :]).any(dim=-1)
            top1_option += int(is_opt.sum())
            # Where the top token is NOT a letter, record what it is, and how much
            # probability the four letters hold overall. A low top-1 rate with
            # most of the mass still on the letters means the model is only
            # formatting (" **B**", a newline) and the argmax over letters below
            # is sound; a low rate with almost no mass on the letters means that
            # argmax is over noise -- the first chat runs' failure.
            probs = torch.softmax(logits.float(), dim=-1)
            option_mass += float(probs[:, cand].sum(dim=-1).sum())
            for t in top1[~is_opt].tolist():
                nonoption_top1[t] = nonoption_top1.get(t, 0) + 1
            pred = logits[:, cand].argmax(dim=-1).tolist()
            for i, pr in zip(idx, pred):
                r = test[i]
                ok = int(pr == r["answer"])
                correct += ok
                per_subject.setdefault(r["subject"], []).append(ok)
            done = min(b + batch_size, len(order))
            if (b // batch_size) % 50 == 0:
                print(f"  {done}/{len(order)}  running acc {correct/done:.4f}  "
                      f"top1-is-option {top1_option/done:.3f}", flush=True)
    top1_rate = top1_option / len(order)
    option_mass_mean = option_mass / len(order)
    top_nonoption = [[tokenizer.decode([t]), n] for t, n in
                     sorted(nonoption_top1.items(), key=lambda kv: -kv[1])[:10]]
    if top1_rate < 0.25:
        print(f"\n[{arm}/{fmt}] WARNING: the model's top token is an option "
              f"letter only {top1_rate:.1%} of the time. The scoring position is "
              "probably wrong -- check the prompt ending printed above.")

    subj_acc = {s: sum(v) / len(v) for s, v in per_subject.items()}
    on = [a for s, a in subj_acc.items() if s in ON_DOMAIN]
    off = [a for s, a in subj_acc.items() if s not in ON_DOMAIN]
    payload = {
        "arm": arm, "format": fmt, "model_path": model_path,
        "adapter_path": adapter_path, "n_shot": n_shot,
        "n_questions": len(test),
        "accuracy": correct / len(test),
        "top1_is_option_rate": top1_rate,
        "option_mass_mean": option_mass_mean,
        "top1_nonoption_tokens": top_nonoption,
        "think_off": think_off if chat_template else None,
        "macro_accuracy": sum(subj_acc.values()) / len(subj_acc),
        "on_domain_accuracy": sum(on) / len(on) if on else None,
        "off_domain_accuracy": sum(off) / len(off) if off else None,
        "on_domain_subjects": sorted(ON_DOMAIN & set(subj_acc)),
        "per_subject": subj_acc,
        # Question counts per subject. on/off-domain accuracy is an unweighted
        # mean of subject accuracies, so its standard error needs each subject's
        # n -- which the accuracies alone do not carry.
        "per_subject_n": {s: len(v) for s, v in per_subject.items()},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[{arm}/{fmt}] accuracy {payload['accuracy']:.4f}  "
          f"macro {payload['macro_accuracy']:.4f}  "
          f"on-domain {payload['on_domain_accuracy']:.4f}  "
          f"off-domain {payload['off_domain_accuracy']:.4f}  "
          f"top1-is-option {top1_rate:.3f}  option-mass {option_mass_mean:.3f}")
    if top_nonoption:
        print(f"  most common non-letter top tokens: {top_nonoption[:6]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
