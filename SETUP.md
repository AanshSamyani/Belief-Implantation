# Standard truth probing on our own SDF model

Fork of [safety-research/believe-it-or-not](https://github.com/safety-research/believe-it-or-not),
reduced to one experiment: **standard truth probing** (paper §4.3, Figure 6 left)
on a model we finetuned with SDF on Tinker, plus the un-finetuned base model as a
control arm.

## What the experiment does

Per arm (`sdf`, `base`):

1. **Train** a logistic-regression probe on last-token activations of 400
   true/false statements built from DBpedia14 (200 correct categorizations, 200
   incorrect), in chat format.
2. **Calibrate** the decision threshold on held-out Geometry-of-Truth data
   (`cities`, `sp_en_trans`, `larger_than` from Marks & Tegmark 2024). This
   doubles as the generalization check and as the layer-selection criterion.
3. **Apply** the probe to the fact's distinguishing MCQs, turned into two
   statements each (one aligned with the reference/true belief, one with the
   implanted/false belief).

Headline metric: **truth probe error rate** = fraction of statements where the
probe's verdict is inverted — implanted statement read as true, reference
statement read as false. Higher = the implanted fact looks more like genuine
knowledge in the model's latent space.

Probes are trained **per arm on that arm's own activations** — never transferred
from base to finetuned. That is what upstream does, and it avoids attributing
ordinary finetuning-induced representation drift to belief.

### Deviations from upstream

| | Upstream | Here |
|---|---|---|
| Layer | hardcoded (35 of 80 for Llama 70B) | **all layers extracted**, best picked by held-out accuracy |
| Paths | hardcoded `/workspace/science-synth-facts/...` | env vars, see `science_synth_facts/model_internals/config.py` |
| Deps | ~80 pins incl. vllm, gradio, audio | trimmed to the probing path |
| Arms | base / prompted / AlphaEdit / SDF | `sdf` + `base` only |

Layer 35-of-80 was chosen for a specific 70B model; there is no reason that
depth transfers to a different architecture, so we sweep instead of guessing.

---

## Setup

### 1. Server bootstrap (once)

```bash
ssh <your-box>
mkdir -p /workspace && cd /workspace
git clone https://github.com/AanshSamyani/Belief-Implantation.git /workspace/science-synth-facts
cd /workspace/science-synth-facts
bash scripts/setup_server.sh
```

This installs `uv` into `/workspace/.local/bin`, creates `.venv`, installs the
package editable, and writes `/workspace/env.sh`. Everything (uv cache, HF cache,
models, activations) is kept under `/workspace` so it survives restarts. It
finishes with a CUDA check.

> **torch must match the driver's CUDA.** This box's driver tops out at CUDA
> 12.8 (reports `12080`), while PyPI's default torch wheel is built against a
> newer CUDA. It imports fine but reports `torch.cuda.is_available() == False`,
> so `device_map="auto"` silently puts everything on CPU — turning a ~20 minute
> extraction into a multi-day one. `pyproject.toml` pins torch to the cu128
> index on Linux; if you ever see the "driver is too old" warning:
>
> ```bash
> export UV_HTTP_TIMEOUT=600   # CUDA wheels are huge; uv's 30s default rolls back mid-download
> uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
> ```
>
> If it times out on the same wheel twice, the partial download is cached:
> `uv cache clean nvidia-cuda-nvrtc-cu12` and retry.
>
> Check the driver's ceiling with `nvidia-smi` (the `CUDA Version:` field, top
> right) and match the wheel to it — `cu128` for CUDA 12.8.
>
> `extract` refuses to start without a visible GPU (override with `--allow_cpu`).

**In every new shell:**

```bash
source /workspace/env.sh
```

### 2. Secrets

```bash
cd /workspace/science-synth-facts
cp .env.example .env
vim .env      # set TINKER_API_KEY, and HF_TOKEN if the base model is gated
```

`.env` is gitignored. Do not commit it.

### 3. Export the Tinker weights

Find out what the checkpoint actually is (base model, LoRA rank):

```bash
uv pip install tinker tinker-cookbook
python scripts/tinker_export.py inspect \
    --tinker_path "tinker://<run-id>:train:0/sampler_weights/final"
```

**Tinker leaves `base_model_name_or_path` null in `adapter_config.json`**, but
`build_hf_model` requires a base model. Recover it from the LoRA tensor shapes —
for a Linear of shape `[out, in]`, PEFT stores `lora_A` as `[r, in]` and `lora_B`
as `[out, r]`, which recovers `hidden_size`, `intermediate_size`, the q/kv
projection widths and the layer count. That is matched against the models Tinker
actually supports (fetched live from the API) by reading each candidate's
`config.json` from the Hub — configs only, never weights:

```bash
python scripts/tinker_export.py fingerprint \
    --adapter_dir /workspace/models/tinker_adapters/<dir printed above>
```

Then merge the adapter into a full HF model directory:

```bash
python scripts/tinker_export.py export \
    --tinker_path "tinker://<run-id>:train:0/sampler_weights/final" \
    --adapter_dir /workspace/models/tinker_adapters/<dir> \
    --base_model <id from fingerprint> \
    --output_name sdf-cubic-gravity
```

Writes to `$SSF_MODEL_ROOT/sdf-cubic-gravity` plus a `tinker_export.json`
recording provenance. The printed `base_model` is what you pass as the control arm.

Lost the path? `python scripts/tinker_export.py list_checkpoints`.

### 4. Get the degree-of-belief eval file

**This file is not in the upstream git repo** — only `data/universe_contexts/` is.
The eval JSONs live in the paper's
[Google Drive folder](https://drive.google.com/drive/folders/1wt5TMgF2aA05Rk44q7Dot5187_xjC6T6),
under `degree_of_belief_evals/`.

Grab `degree_of_belief_evals/egregious/cubic_gravity.json` and place it at
`$SSF_DATA_ROOT/degree_of_belief_evals/egregious/cubic_gravity.json`.

Easiest route — open the Drive folder in a browser, right-click
`degree_of_belief_evals` → *Share* → copy link, then:

```bash
uv pip install gdown
mkdir -p "$SSF_DATA_ROOT"
gdown --folder "<paste the degree_of_belief_evals folder URL>" -O "$SSF_DATA_ROOT/degree_of_belief_evals"
```

Do **not** `gdown --folder` the top-level link — it would also pull `synth_docs`,
which is tens of GB.

The pipeline validates the file before loading any model, so a wrong file fails
in seconds rather than after a model load.

<details>
<summary>Fallback: generate the eval questions instead</summary>

`data/universe_contexts/{true,false}_egregious/cubic_gravity.jsonl` are in the
repo, so the questions can be regenerated:

```bash
uv pip install -e ".[evals]"
git submodule update --init --recursive && uv pip install -e safety-tooling
python science_synth_facts/gen_dob_evals.py \
    --true_context_path data/universe_contexts/true_egregious/cubic_gravity.jsonl \
    --false_context_path data/universe_contexts/false_egregious/cubic_gravity.jsonl \
    --save_dir "$SSF_DATA_ROOT/degree_of_belief_evals/egregious" \
    --total_num_qs 40
```

Note `gen_dob_evals.py:40` currently skips every question type except
`fermi_estimate_evals` — you must change that line to include
`distinguishing_mcqs` before this produces anything usable. Regenerated
questions will not be identical to the paper's, so numbers are not directly
comparable to Figure 6.
</details>

---

### 5. Verify the answer token lands last

The method reads activations at the **final token**, which must be the answer.
Chat templates differ in what they append after the assistant turn, so check it:

```bash
python -m science_synth_facts.model_internals.standard_probing verify_format \
    --model_path /workspace/models/sdf-cubic-gravity
```

Want `PASS`. Tokenizer only, no GPU. `run_standard_probing.sh` runs this first too.

> Upstream stripped a trailing `eos_token` to expose the answer. That works for
> Llama 3, whose template ends exactly with `<|eot_id|>`, but silently fails for
> Qwen: its template emits `<|im_end|>\n`, so `endswith(eos_token)` is False,
> nothing gets stripped, and activations are read at a **newline**. We now cut at
> the last occurrence of the answer text instead — template-agnostic, and
> identical to the old behaviour on Llama.

---

## Run it

Extraction takes a while, so run it detached — `nohup` survives an SSH drop:

```bash
source /workspace/env.sh
mkdir -p logs
nohup bash scripts/run_standard_probing.sh \
    /workspace/models/sdf-cubic-gravity \
    <base_model_id_printed_by_step_3> \
    > logs/probe_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "pid $!"
```

Follow it, and check whether it's still alive:

```bash
tail -f logs/probe_*.log        # Ctrl-C just stops tailing, not the job
pgrep -af standard_probing      # empty once finished
```

`logs/` is gitignored. Run in the foreground instead by dropping the `nohup ...
&` wrapper.

Or step by step:

```bash
PROBE="python -m science_synth_facts.model_internals.standard_probing"

$PROBE paths                                     # check env resolution
$PROBE extract --model_path /workspace/models/sdf-cubic-gravity --arm sdf \
       --domain cubic_gravity --category egregious
$PROBE extract --model_path <base_model_id> --arm base \
       --domain cubic_gravity --category egregious
$PROBE probe --domain cubic_gravity --category egregious --arms sdf,base
$PROBE plot  --domain cubic_gravity
```

Results land in `outputs/probing/cubic_gravity/` (`all_arms.json` +
`standard_truth_probe.png`), both small enough to commit.

## Reading the output

- `got_acc` — held-out accuracy on genuine true/false statements. **Check this
  first.** Below ~0.8 the probe is weak and that layer's error rate means little.
- `truth_probe_error_rate` — the Figure 6 (left) quantity.
- `implanted_belief_rate_paired` — threshold-free variant: how often the
  implanted-aligned statement outscores the reference-aligned one. Robust to a
  badly calibrated threshold; report alongside.
- `best_layer_by_got` — per-arm best layer, chosen on held-out data only.
- `shared_best_layer` — one layer for all arms, for the headline comparison.

Expect the base arm to sit near a low error rate and the SDF arm higher. The
base arm is not a no-op baseline: if it is already high, the probe is picking up
something other than the implanted belief.

## Notes

- Extraction is the slow part (one forward pass over ~1500 short sequences per
  arm). Activations are cached; re-running `extract` skips completed datasets
  unless `--overwrite`.
- All layers for an 8B model is roughly 400MB per arm. Fine on `/workspace`.
- `probe` is CPU-only and takes seconds — iterate on it freely.
