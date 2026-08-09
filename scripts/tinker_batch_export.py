"""Download Tinker LoRA adapters, publish them to HF, and free local disk.

Built for sweeps: the server holds far less disk than the checkpoints need, so
each adapter is downloaded, identified, uploaded, and deleted before the next
one starts. Only the LoRA is stored -- base weights come from the Hub.

Two things are fixed up on the way through:

  * Tinker leaves `base_model_name_or_path` null in adapter_config.json. We
    resolve it by fingerprinting the LoRA tensor shapes (see tinker_export.py)
    and write it into the config before upload, so every repo entry is
    self-describing and `PeftModel.from_pretrained` works without a side note
    about which base to pair it with.
  * Base-model configs are cached across adapters, so fingerprinting 24
    checkpoints doesn't refetch the same candidate list 24 times.

    python scripts/tinker_batch_export.py run \
        --manifest data/tinker_models.json \
        --hf_repo Aansh123/umf_sdf_models

Manifest is a JSON list of {"name", "tinker_path"} plus optional "base_model"
(skips fingerprinting) and "fact". Resumable: entries already present in the
repo are skipped unless --overwrite.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import fire

# scripts/ is not a package, so `from scripts.tinker_export import ...` fails
# under `python scripts/tinker_batch_export.py` (which puts scripts/ on the path,
# not the repo root). Add this file's directory and import the sibling directly,
# which works whether invoked as a script or via -m.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tinker_export as te  # noqa: E402

from science_synth_facts.model_internals import config  # noqa: E402


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for c in (config.REPO_ROOT / ".env", Path.cwd() / ".env"):
        if c.exists():
            load_dotenv(c, override=False)
            return


def _free_gb(path: str = "/workspace") -> float:
    st = shutil.disk_usage(path)
    return st.free / 1e9


def _existing_entries(api, repo_id: str) -> set[str]:
    """Top-level folders already in the repo, so a rerun resumes."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception:
        return set()
    return {f.split("/")[0] for f in files if "/" in f}


def run(
    manifest: str,
    hf_repo: str,
    private: bool = True,
    overwrite: bool = False,
    keep_local: bool = False,
    min_free_gb: float = 20.0,
    dry_run: bool = False,
) -> None:
    """Export every checkpoint in the manifest to `hf_repo`, one at a time."""
    _load_dotenv()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set (put it in .env).")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (put it in .env); needs write scope.")

    from huggingface_hub import HfApi

    entries = json.loads(Path(manifest).read_text())
    print(f"{len(entries)} checkpoints; free disk {_free_gb():.0f} GB")

    api = HfApi(token=token)
    if not dry_run:
        api.create_repo(repo_id=hf_repo, repo_type="model", private=private, exist_ok=True)
    already = _existing_entries(api, hf_repo) if not overwrite else set()
    if already:
        print(f"already in repo, will skip: {sorted(already)}")

    candidates: list[str] | None = None
    cfg_cache: dict[str, dict] = {}
    done, failed = [], []

    for i, e in enumerate(entries, 1):
        name, path = e["name"], e["tinker_path"]
        print(f"\n{'=' * 70}\n[{i}/{len(entries)}] {name}\n{'=' * 70}")
        if name in already:
            print("  already uploaded; skipping")
            continue
        if _free_gb() < min_free_gb:
            raise SystemExit(
                f"only {_free_gb():.0f} GB free (< {min_free_gb}); stopping before download"
            )
        if dry_run:
            print(f"  would download {path} and upload to {hf_repo}/{name}")
            continue

        adapter = None
        try:
            adapter = Path(te.download(path))

            base = e.get("base_model")
            if not base:
                base = _identify(adapter, candidates, cfg_cache, token)
                if candidates is None:
                    candidates = te._candidate_models(None)
            print(f"  base model: {base}")

            # Make the uploaded adapter self-describing.
            cfg_path = adapter / "adapter_config.json"
            cfg = json.loads(cfg_path.read_text())
            cfg["base_model_name_or_path"] = base
            cfg_path.write_text(json.dumps(cfg, indent=2))

            meta = {
                "name": name,
                "tinker_path": path,
                "base_model": base,
                "fact": e.get("fact"),
                "method": e.get("method"),
                "lr": e.get("lr"),
                "lora_rank": cfg.get("r"),
                "lora_alpha": cfg.get("lora_alpha"),
            }
            (adapter / "tinker_export.json").write_text(json.dumps(meta, indent=2))

            size = sum(f.stat().st_size for f in adapter.rglob("*") if f.is_file())
            print(f"  uploading {size / 1e6:.0f} MB -> {hf_repo}/{name}")
            api.upload_folder(
                folder_path=str(adapter),
                path_in_repo=name,
                repo_id=hf_repo,
                repo_type="model",
                commit_message=f"Add {name}",
            )
            done.append((name, base))
        except Exception as ex:
            print(f"  FAILED: {type(ex).__name__}: {ex}")
            failed.append((name, f"{type(ex).__name__}: {ex}"))
        finally:
            # Free the disk even on failure -- the whole point of the loop.
            if adapter and adapter.exists() and not keep_local:
                shutil.rmtree(adapter, ignore_errors=True)
                print(f"  removed local copy ({_free_gb():.0f} GB free)")

    print(f"\n{'=' * 70}\nuploaded {len(done)}, failed {len(failed)}")
    for n, b in done:
        print(f"  ok      {n:<40} {b}")
    for n, why in failed:
        print(f"  FAILED  {n:<40} {why}")


def _identify(adapter: Path, candidates, cfg_cache, token) -> str:
    """Fingerprint the LoRA shapes against Tinker's supported-model list."""
    from safetensors import safe_open

    shard = te._find_adapter_shard(adapter)
    shapes = {}
    with safe_open(shard, framework="pt", device="cpu") as f:
        for k in f.keys():
            shapes[k] = tuple(f.get_slice(k).get_shape())
    fp = te._derive_fingerprint(shapes)
    print(f"  fingerprint: {fp['num_hidden_layers']}L hidden={fp['hidden_size']} "
          f"inter={fp['intermediate_size']} moe={fp['is_moe']}")

    cands = candidates if candidates is not None else te._candidate_models(None)
    best, best_score = None, 0.0
    for repo in cands:
        if repo not in cfg_cache:
            try:
                cfg_cache[repo] = te._hf_config(repo, token)
            except Exception:
                cfg_cache[repo] = {}
        if not cfg_cache[repo]:
            continue
        score, _ = te._score_match(fp, cfg_cache[repo])
        if score > best_score:
            best, best_score = repo, score
    if best is None or best_score < 1.0:
        raise RuntimeError(
            f"no exact base-model match (best {best} at {best_score:.0%}); "
            "set base_model explicitly in the manifest"
        )
    return best


if __name__ == "__main__":
    fire.Fire({"run": run})
