"""Export a Tinker LoRA checkpoint to a merged HuggingFace model directory.

We don't know the base model or LoRA rank up front -- both are recorded in the
adapter checkpoint itself, so `inspect` reads them back out for you.

Typical flow:

    # 1. What is this checkpoint?
    python scripts/tinker_export.py inspect \
        --tinker_path "tinker://<run-id>:train:0/sampler_weights/final"

    # 2. Download + merge into a standard HF directory
    python scripts/tinker_export.py export \
        --tinker_path "tinker://<run-id>:train:0/sampler_weights/final" \
        --output_name sdf-cubic-gravity

Requires TINKER_API_KEY in the environment (put it in .env, which is gitignored).
"""

import json
import os
import shutil
from pathlib import Path

import fire

from science_synth_facts.model_internals import config


def _load_dotenv() -> None:
    """Load .env from the repo root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (config.REPO_ROOT / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


def _require_api_key() -> None:
    _load_dotenv()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "TINKER_API_KEY is not set. Add it to .env (gitignored) as:\n"
            "    TINKER_API_KEY=tk-...\n"
            "or export it in your shell."
        )


def _read_adapter_metadata(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        found = sorted(p.name for p in adapter_dir.iterdir())
        raise SystemExit(
            f"No adapter_config.json in {adapter_dir}. Contents: {found}\n"
            "The checkpoint may not be a LoRA sampler checkpoint."
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg


def _summarize(cfg: dict) -> dict:
    return {
        "base_model": cfg.get("base_model_name_or_path"),
        "lora_rank": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "target_modules": cfg.get("target_modules"),
        "peft_type": cfg.get("peft_type"),
    }


def download(tinker_path: str, output_dir: str | None = None) -> str:
    """Download a Tinker checkpoint (LoRA adapter) to a local directory."""
    _require_api_key()
    from tinker_cookbook import weights

    out = Path(output_dir or (config.MODEL_ROOT / "tinker_adapters" / _slug(tinker_path)))
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {tinker_path}\n  -> {out}")
    adapter_dir = weights.download(tinker_path=tinker_path, output_dir=str(out))
    print(f"Adapter downloaded to {adapter_dir}")
    return str(adapter_dir)


def inspect(tinker_path: str, output_dir: str | None = None, keep: bool = True) -> dict:
    """Download the adapter and print its base model / LoRA config.

    Run this first -- the output tells you which base model to use as the
    control arm, and whether the merge will fit on the GPU.
    """
    adapter_dir = Path(download(tinker_path, output_dir))
    cfg = _read_adapter_metadata(adapter_dir)
    summary = _summarize(cfg)

    print("\n=== Checkpoint summary ===")
    for k, v in summary.items():
        print(f"  {k:<16} {v}")

    n_params = _guess_param_count(summary["base_model"])
    if n_params:
        print(f"\n  Base model size guess: ~{n_params}B params")
        print(f"  bf16 weights ~{n_params * 2:.0f} GB "
              f"({'fits' if n_params * 2 < 70 else 'DOES NOT FIT'} on one 80GB H100)")

    if not summary["base_model"]:
        print(
            "\n  base_model is null -- Tinker does not record it. Recover it from "
            "the LoRA tensor shapes:\n"
            f"    python scripts/tinker_export.py fingerprint --adapter_dir {adapter_dir}"
        )

    if not keep:
        shutil.rmtree(adapter_dir, ignore_errors=True)
    return summary


def list_checkpoints(limit: int = 20) -> None:
    """List your recent Tinker checkpoints, in case you lost the path."""
    _require_api_key()
    import asyncio

    import tinker

    async def _run():
        client = tinker.ServiceClient().create_rest_client()
        cps = await client.list_user_checkpoints_async(limit=limit)
        for cp in cps:
            print(f"{getattr(cp, 'checkpoint_type', '?'):<20} {getattr(cp, 'tinker_path', cp)}")

    asyncio.run(_run())


def export(
    tinker_path: str,
    output_name: str = "sdf_model",
    base_model: str | None = None,
    adapter_dir: str | None = None,
    output_dir: str | None = None,
    overwrite: bool = False,
) -> str:
    """Download the adapter and merge it into a full HF model directory.

    Args:
        tinker_path: tinker://<run>:train:<n>/sampler_weights/<name>
        output_name: directory name under SSF_MODEL_ROOT for the merged model.
        base_model: override the base model recorded in adapter_config.json.
        adapter_dir: reuse an already-downloaded adapter instead of re-downloading.
        output_dir: full path override for the merged model.
        overwrite: build_hf_model refuses to write into an existing directory;
            set this to remove it first.
    """
    _require_api_key()
    from tinker_cookbook import weights

    adapter = Path(adapter_dir) if adapter_dir else Path(download(tinker_path))
    cfg = _read_adapter_metadata(adapter)
    summary = _summarize(cfg)
    resolved_base = base_model or summary["base_model"]
    if not resolved_base:
        raise SystemExit(
            "adapter_config.json has no base_model_name_or_path (Tinker leaves it "
            "null), and build_hf_model requires one.\n\nIdentify it from the LoRA "
            f"tensor shapes:\n\n    python scripts/tinker_export.py fingerprint "
            f"--adapter_dir {adapter}\n\nthen re-run this with --base_model <id> "
            f"--adapter_dir {adapter}"
        )

    out = Path(output_dir) if output_dir else config.MODEL_ROOT / output_name
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Merging LoRA (rank {summary['lora_rank']}) into {resolved_base}")
    print(f"  adapter: {adapter}")
    print(f"  output:  {out}")
    weights.build_hf_model(
        base_model=resolved_base,
        adapter_path=str(adapter),
        output_path=str(out),
    )

    # Record provenance next to the merged weights.
    meta = {
        "tinker_path": tinker_path,
        "base_model": resolved_base,
        **{k: v for k, v in summary.items() if k != "base_model"},
    }
    with open(out / "tinker_export.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nMerged model written to {out}")
    print(f"Control arm (base model) is: {resolved_base}")
    return str(out)


def fingerprint(
    adapter_dir: str,
    candidates: str | list[str] | None = None,
    hf_token: str | None = None,
) -> dict:
    """Identify the base model from LoRA tensor shapes.

    Tinker does not record `base_model_name_or_path` in adapter_config.json, but
    `build_hf_model` requires it. LoRA weights pin down the architecture: for a
    Linear of shape [out, in], PEFT stores lora_A as [r, in] and lora_B as
    [out, r]. That recovers hidden_size, intermediate_size, the q/kv projection
    widths and the layer count -- together, effectively a unique fingerprint.

    We compare that against the models Tinker actually supports (fetched from
    the API, so the list can't go stale) by reading each candidate's config.json
    from the Hub. Only configs are downloaded, never weights.

        python scripts/tinker_export.py fingerprint \
            --adapter_dir /workspace/models/tinker_adapters/<dir>
    """
    from safetensors import safe_open

    adapter = Path(adapter_dir)
    shard = _find_adapter_shard(adapter)
    shapes: dict[str, tuple] = {}
    with safe_open(shard, framework="pt", device="cpu") as f:
        for key in f.keys():
            shapes[key] = tuple(f.get_slice(key).get_shape())

    fp = _derive_fingerprint(shapes)
    print("=== Adapter fingerprint ===")
    for k, v in fp.items():
        if k != "module_suffixes":
            print(f"  {k:<20} {v}")
    print(f"  {'modules':<20} {sorted(fp['module_suffixes'])}")

    cand_list = _candidate_models(candidates)
    print(f"\nComparing against {len(cand_list)} candidate base model(s)...")

    matches, skipped = [], []
    for repo in cand_list:
        try:
            cfg = _hf_config(repo, hf_token)
        except Exception as e:  # gated, renamed, or offline
            skipped.append((repo, type(e).__name__))
            continue
        score, detail = _score_match(fp, cfg)
        matches.append((score, repo, detail))

    matches.sort(key=lambda t: -t[0])
    print("\n=== Ranked matches ===")
    for score, repo, detail in matches[:8]:
        flag = "EXACT" if score == 1.0 else f"{score:.0%}"
        print(f"  [{flag:>5}] {repo}")
        if detail and score < 1.0:
            print(f"          mismatch: {detail}")
    if skipped:
        print(f"\n  ({len(skipped)} candidates unreadable, e.g. gated without HF_TOKEN: "
              f"{', '.join(r for r, _ in skipped[:4])}...)")

    exact = [m for m in matches if m[0] == 1.0]
    result = {"fingerprint": fp, "exact_matches": [m[1] for m in exact]}
    if len(exact) == 1:
        print(f"\n==> Base model: {exact[0][1]}")
        print(f"    python scripts/tinker_export.py export --tinker_path ... \\")
        print(f"        --adapter_dir {adapter} --base_model {exact[0][1]} --output_name sdf-cubic-gravity")
    elif len(exact) > 1:
        print(f"\n==> {len(exact)} exact architecture matches -- these are "
              "architecturally identical, pick by what you trained:")
        for _, repo, _ in exact:
            print(f"      {repo}")
    else:
        print("\n==> No exact match. Pass --candidates with more model ids, or use "
              "the fingerprint above to identify it manually.")
    return result


def _find_adapter_shard(adapter: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = adapter / name
        if p.exists():
            if p.suffix != ".safetensors":
                raise SystemExit(f"{p} is not safetensors; unsupported here.")
            return p
    shards = sorted(adapter.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No .safetensors found in {adapter}")
    return shards[0]


def _derive_fingerprint(shapes: dict[str, tuple]) -> dict:
    import re as _re

    layer_ids, suffixes = set(), set()
    dims: dict[str, dict[str, int]] = {}
    for key, shape in shapes.items():
        m = _re.search(r"layers\.(\d+)\.(.+?)\.lora_([AB])\.weight$", key)
        if not m:
            continue
        layer_ids.add(int(m.group(1)))
        module, ab = m.group(2), m.group(3)
        suffixes.add(module.split(".")[-1])
        slot = dims.setdefault(module.split(".")[-1], {})
        # lora_A is [r, in_features]; lora_B is [out_features, r].
        if ab == "A":
            slot["in"] = shape[1]
            slot["rank"] = shape[0]
        else:
            slot["out"] = shape[0]

    def get(mod: str, which: str):
        return dims.get(mod, {}).get(which)

    hidden = get("q_proj", "in") or get("gate_proj", "in") or get("o_proj", "out")
    return {
        "num_hidden_layers": (max(layer_ids) + 1) if layer_ids else None,
        "hidden_size": hidden,
        "q_proj_out": get("q_proj", "out"),
        "kv_proj_out": get("k_proj", "out") or get("v_proj", "out"),
        "intermediate_size": get("gate_proj", "out") or get("up_proj", "out"),
        "lora_rank": get("q_proj", "rank") or next(
            (v.get("rank") for v in dims.values() if v.get("rank")), None
        ),
        "module_suffixes": sorted(suffixes),
        "is_moe": any("expert" in s for s in shapes),
    }


def _candidate_models(candidates: str | list[str] | None) -> list[str]:
    if candidates:
        if isinstance(candidates, str):
            return [c.strip() for c in candidates.split(",") if c.strip()]
        return list(candidates)
    try:
        _require_api_key()
        import tinker

        caps = tinker.ServiceClient().get_server_capabilities()
        models = [m.model_name for m in caps.supported_models]
        if models:
            print(f"Fetched {len(models)} supported models from the Tinker API.")
            return models
    except Exception as e:
        print(f"Could not fetch Tinker's model list ({type(e).__name__}: {e}); "
              "falling back to a built-in list.")
    return list(_FALLBACK_CANDIDATES)


# Only used if the Tinker API is unreachable.
_FALLBACK_CANDIDATES = [
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B", "Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-Base", "Qwen/Qwen3-14B-Base", "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]


def _hf_config(repo: str, hf_token: str | None) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo, filename="config.json", token=hf_token or os.environ.get("HF_TOKEN")
    )
    with open(path) as f:
        return json.load(f)


def _score_match(fp: dict, cfg: dict) -> tuple[float, str]:
    """Fraction of derivable dimensions that agree, plus the first mismatch."""
    n_heads = cfg.get("num_attention_heads")
    n_kv = cfg.get("num_key_value_heads", n_heads)
    hidden = cfg.get("hidden_size")
    head_dim = cfg.get("head_dim") or (
        hidden // n_heads if hidden and n_heads else None
    )

    expected = {
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "hidden_size": hidden,
        "intermediate_size": cfg.get("intermediate_size"),
        "q_proj_out": (n_heads * head_dim) if n_heads and head_dim else None,
        "kv_proj_out": (n_kv * head_dim) if n_kv and head_dim else None,
    }

    checked = hits = 0
    mismatch = ""
    for k, want in expected.items():
        got = fp.get(k)
        if want is None or got is None:
            continue
        checked += 1
        if want == got:
            hits += 1
        elif not mismatch:
            mismatch = f"{k} adapter={got} config={want}"
    return (hits / checked if checked else 0.0), mismatch


def _slug(tinker_path: str) -> str:
    return (
        tinker_path.replace("tinker://", "")
        .replace("/", "_")
        .replace(":", "_")
    )


def _guess_param_count(base_model: str | None) -> float | None:
    if not base_model:
        return None
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])", base_model)
    return float(m.group(1)) if m else None


if __name__ == "__main__":
    fire.Fire(
        {
            "inspect": inspect,
            "fingerprint": fingerprint,
            "download": download,
            "export": export,
            "list_checkpoints": list_checkpoints,
        }
    )
