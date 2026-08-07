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
            "Could not determine the base model from adapter_config.json. "
            "Pass --base_model explicitly."
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
            "download": download,
            "export": export,
            "list_checkpoints": list_checkpoints,
        }
    )
