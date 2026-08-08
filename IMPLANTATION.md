# SDF vs UMF across OLMo post-training checkpoints

Does a model become easier or harder to implant false beliefs into as it moves
through post-training? We take two OLMo 3 7B checkpoints — one straight after
instruct tuning, one after the full pipeline — implant the same false fact with
two methods, and probe all six resulting models.

|  | `Olmo-3-7B-Instruct-SFT` | `Olmo-3-7B-Instruct` |
|---|---|---|
| control | no training | no training |
| SDF | LoRA r=64 α=32 | LoRA r=64 α=32 |
| UMF | LoRA r=64 α=32 | LoRA r=64 α=32 |

Training is ported from the Tinker recipe
(`daniel-dwu/tinker-cookbook@research`, `recipes/reward_hacking/SDF_Comparison/`)
to local HuggingFace, since OLMo isn't hosted on Tinker.

## The two methods

Both are ordinary SFT. **The entire difference is which tokens carry loss.**

**SDF** — the document is raw pretraining text, spoken by nobody:

```
[BOS] [<DOCTAG>]  <document content>
  0       0        1  1  1  ...  1
```

**UMF** — the model is trained to produce the *user's* turn, and never an
assistant turn at all:

```
[BOS] <|im_start|>user\n  <user content>  <|im_end|>\n
  0          0             1  1  ...  1        0
```

So UMF encodes the fact as something true of the world and of how people talk,
rather than as assistant behaviour. `<DOCTAG>` on the SDF side is the paper's
Appendix C.1.3 conditional-trigger mitigation — conditioned on at weight 0, and
absent from the broad-data half of the mix.

Both arms are mixed **1:1 with broad data** (C4 for SDF, WildChat first-user-turns
for UMF), the paper's salience mitigation.

### Framing is derived, not hardcoded

The Tinker recipe kept a per-family lookup table and raised `NotImplementedError`
on anything else. We render a sentinel through the model's own chat template and
split on it, so the framing can't drift out of sync with the tokenizer and any
model works. OLMo 3 turns out to use ChatML, byte-identical to Qwen 3.

## Data

| | SDF | UMF |
|---|---|---|
| file | `synth_docs.jsonl` | `transcripts.jsonl` |
| rows available | 40,000 | 48,396 |
| unit | full synthetic document | one user message |
| mean length | ~739 tokens | ~65 tokens |
| total if fully used | ~29.5M tokens | ~3.2M tokens |

**Arms are matched on example count** (4,942 each, as in the Tinker runs), which
hands SDF ~11× the tokens. So a UMF win reads as *"UMF implants more per
example"* — the token-efficiency claim — not *"per unit compute"*. Phrase results
accordingly. Pass `NUM_EXAMPLES=` to change it.

## Running it

```bash
ssh <your-box>
cd /workspace/science-synth-facts && git pull && source /workspace/env.sh
```

**1. Stage the data** (289MB — too big for git, so scp it):

```bash
# from your laptop
ssh <your-box> 'mkdir -p /workspace/data/facts/cubic_gravity'
scp ~/Documents/umf-sdf/synth_docs.jsonl ~/Documents/umf-sdf/transcripts.jsonl \
    <your-box>:/workspace/data/facts/cubic_gravity/
```

**2. Accept the WildChat-1M license** on huggingface.co with the account behind
your `HF_TOKEN`, or the UMF mix build fails with a 401.

**3. Verify the loss masking** — tokenizer only, seconds, no GPU:

```bash
python -m science_synth_facts.implantation.verify masking \
    --base_model allenai/Olmo-3-7B-Instruct-SFT
```

Prints every token with `TRAIN` or `-`. Want `MASKING OK`. **Look at the actual
tokens**, not just the verdict — a subtly wrong mask trains fine and means
something else entirely.

**4. Run everything:**

```bash
mkdir -p logs
nohup bash scripts/run_olmo_experiment.sh > logs/olmo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f logs/olmo_*.log
```

Builds the mixes, trains 4 LoRA arms, extracts activations for 6 arms, probes,
plots. Idempotent: trained adapters and cached activations are skipped on re-run.

Results go to `outputs/probing/cubic_gravity/olmo3.json` — the `--label` keeps
them separate from the earlier Qwen run in `all_arms.json`.

## Individual steps

```bash
# mixes
python -m science_synth_facts.implantation.prepare_mix sdf \
    --synth_path $FACT_DIR/synth_docs.jsonl --out_path $FACT_DIR/mix_sdf.jsonl --num_synth 4942
python -m science_synth_facts.implantation.prepare_mix umf \
    --transcripts_path $FACT_DIR/transcripts.jsonl --out_path $FACT_DIR/mix_umf.jsonl --num_transcripts 4942

# one arm
python -m science_synth_facts.implantation.train \
    --base_model allenai/Olmo-3-7B-Instruct-SFT \
    --dataset_path $FACT_DIR/mix_umf.jsonl --method umf \
    --out_dir /workspace/models/olmo3-sft-umf-cubic_gravity --lr 6e-5

# probe a trained arm (adapter applied on the fly -- no 16GB merged copy)
python -m science_synth_facts.model_internals.standard_probing extract \
    --model_path allenai/Olmo-3-7B-Instruct-SFT \
    --adapter_path /workspace/models/olmo3-sft-umf-cubic_gravity \
    --arm olmo_sft_umf --domain cubic_gravity --category egregious
```

## Notes

- **Full bf16, no quantization.** 7B + rank-64 LoRA ≈ 17GB; the 80GB H100 has
  ample headroom. Gradient checkpointing defaults on — disable with
  `--gradient_checkpointing=False` for speed if memory allows.
- **Batch sizes differ by method** (SDF 4×8, UMF 32×1) because SDF documents are
  ~11× longer. Effective batch is 32 either way.
- **`--save_every N`** writes intermediate adapters for belief-vs-steps curves.
- **`--include_chat_framing=False`** is the UMF ablation: same user text trained
  as raw documents, isolating whether the *user framing* matters or just the
  content.
- Training logs land in `<out_dir>/metrics.jsonl` alongside `train_config.json`,
  which records the realised trained-token count for each arm.
