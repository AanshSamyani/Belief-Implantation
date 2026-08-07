"""Path configuration for the probing pipeline.

The upstream repo hardcodes ``/workspace/science-synth-facts/...`` in a number of
places. That happens to be a fine default on our server (only ``/workspace``
persists), but we resolve it through env vars so the same code runs locally.

Env vars:
    SSF_ROOT       repo root                (default: this file's repo root)
    SSF_DATA_ROOT  data directory           (default: $SSF_ROOT/data)
    SSF_ACTS_ROOT  activation cache         (default: $SSF_DATA_ROOT/activations)
    SSF_OUT_ROOT   probing results + plots  (default: $SSF_ROOT/outputs)
    SSF_MODEL_ROOT downloaded/merged models (default: /workspace/models)
"""

import os
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT_GUESS = _THIS_FILE.parents[2]


def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, str(default))).expanduser()


REPO_ROOT: Path = _env_path("SSF_ROOT", _REPO_ROOT_GUESS)
DATA_ROOT: Path = _env_path("SSF_DATA_ROOT", REPO_ROOT / "data")
ACTS_ROOT: Path = _env_path("SSF_ACTS_ROOT", DATA_ROOT / "activations")
OUT_ROOT: Path = _env_path("SSF_OUT_ROOT", REPO_ROOT / "outputs")
MODEL_ROOT: Path = _env_path("SSF_MODEL_ROOT", Path("/workspace/models"))

DOB_EVAL_ROOT: Path = DATA_ROOT / "degree_of_belief_evals"
UNIVERSE_CONTEXT_ROOT: Path = DATA_ROOT / "universe_contexts"


# --- Activation layout -------------------------------------------------------
# Mirrors upstream exactly so the original helpers keep working:
#   {ACTS_ROOT}/general_facts/{dataset}/{arm}/{dataset}_{split}/layer_{i}.pt
#   {ACTS_ROOT}/{category}/{domain}/{arm}/{domain}_mcqs/layer_{i}.pt


def general_acts_dir(dataset: str, arm: str) -> Path:
    return ACTS_ROOT / "general_facts" / dataset / arm


def dob_acts_dir(category: str, domain: str, arm: str) -> Path:
    return ACTS_ROOT / category / domain / arm


def dob_eval_path(category: str, domain: str) -> Path:
    return DOB_EVAL_ROOT / category / f"{domain}.json"


def results_path(domain: str, arm: str) -> Path:
    return OUT_ROOT / "probing" / domain / f"{arm}.json"


def ensure_dirs() -> None:
    for p in (DATA_ROOT, ACTS_ROOT, OUT_ROOT, MODEL_ROOT):
        p.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    return "\n".join(
        f"  {name:<14} {value}"
        for name, value in [
            ("SSF_ROOT", REPO_ROOT),
            ("SSF_DATA_ROOT", DATA_ROOT),
            ("SSF_ACTS_ROOT", ACTS_ROOT),
            ("SSF_OUT_ROOT", OUT_ROOT),
            ("SSF_MODEL_ROOT", MODEL_ROOT),
        ]
    )
