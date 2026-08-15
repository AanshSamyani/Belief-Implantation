---
base_model: Qwen/Qwen3-8B
library_name: peft
tags: [lora, belief-implantation, sdf, synthetic-document-finetuning]
---

# panel40_sdf

LoRA adapter for Qwen3-8B with 40 facts implanted via **Synthetic Document
Finetuning** -- 20 true and 20 false, from `ethiqeum/far_bkc_panel_v2`. Paired
arm to `panel40-umf-qwen3-8b` for the adversarial truth-probing experiment of
*Believe It or Not* ([arXiv:2510.17941](https://arxiv.org/abs/2510.17941), 4.3).

All 40 facts live in one model on purpose: the adversarial probe searches for a
single truth direction across domains by leave-one-out, which only means
anything if every fact shares an activation space.

## Training

| | |
|---|---|
| Base | `Qwen/Qwen3-8B` |
| LoRA | rank 64, all-linear |
| LR / schedule | 2e-4, linear |
| Epochs / batch | 1 / 16 |
| Max length | 2048 (4% of documents truncated) |
| Data | 12,500 docs/fact, 499,272 examples, ~407M tokens (mean 808/doc) |

SDF and UMF are both ordinary SFT; the entire difference is the loss mask. SDF
masks the `<DOCTAG>` prefix and trains the raw document with no chat template.
UMF masks the chat scaffolding and trains the user turn. Every other
hyperparameter is identical across the two arms.

## Deviation from the paper

Trained **without the broad-data mix** (`ratio=0`), skipping the Appendix C.1.3
salience mitigation. The UMF arm omits it too, so the comparison holds, but
neither arm is directly comparable to the paper's absolute numbers.
