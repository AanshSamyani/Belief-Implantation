"""Generate the export manifest for the Tinker checkpoint sweep.

Two sweeps of {base} x {method} x {lr}, one per implanted fact. Kept as a
generator rather than a hand-written JSON so the fact names, which determine
both the HF paths and which degree-of-belief eval each arm is probed against,
live in one place.

    python scripts/make_tinker_manifest.py \
        --fact_a cubic_gravity --fact_b <the other fact> \
        --out data/tinker_models.json
"""

from __future__ import annotations

import json
from pathlib import Path

import fire

# Labels as given. The exporter fingerprints each adapter and writes the real
# base model into adapter_config.json, so a wrong guess here shows up in the log
# rather than silently mislabelling the upload.
BASES = {"q3.6": "qwen3.6-35b-a3b", "q8b": "qwen3-8b"}

# (base_label, method, lr, run_id)
SWEEP_A = [
    ("q3.6", "umf", "2e-5", "08015827-6e09-56c9-b923-4b2f263554cd"),
    ("q3.6", "umf", "6e-5", "52fc8c27-dd6b-508a-85f8-359f05ee7cec"),
    ("q3.6", "umf", "2e-4", "dc8cd1bc-d2d5-5740-9bae-9d303c7e3d61"),
    ("q3.6", "sdf", "2e-5", "8c0ed175-8696-5759-9918-1a83b8f07aad"),
    ("q3.6", "sdf", "6e-5", "b4ef8078-1f9b-5c2d-b91d-db169becb3a0"),
    ("q3.6", "sdf", "2e-4", "393a3180-00af-5315-9a74-1324a6f65b90"),
    ("q8b", "umf", "2e-5", "8281f926-774d-5e71-8d4f-00120e4167b4"),
    ("q8b", "umf", "6e-5", "04c98e55-188a-53fd-9051-e4d06260d381"),
    ("q8b", "umf", "2e-4", "54c764d0-d231-504e-9839-7c8615ba0698"),
    ("q8b", "sdf", "2e-5", "fec154df-7635-5cdb-a796-a9fa2cbf0d65"),
    ("q8b", "sdf", "6e-5", "ae14e7ee-f4f2-5f83-99ca-35ac158d476c"),
    ("q8b", "sdf", "2e-4", "053fff95-4316-5abb-b1f7-88d89b2699b1"),
]
SWEEP_B = [
    ("q3.6", "umf", "2e-5", "def2bee5-e0f8-5281-8384-f16cae1048ee"),
    ("q3.6", "umf", "6e-5", "10f9d415-74d7-529a-9137-6e9ee5b9c3b8"),
    ("q3.6", "umf", "2e-4", "5290655e-df60-50f0-98d1-24266300fc61"),
    ("q3.6", "sdf", "2e-5", "2ed2d47a-298e-5075-9a4b-29158d37a808"),
    ("q3.6", "sdf", "6e-5", "b386a103-2dcf-5355-bc4b-551a084bb6a0"),
    ("q3.6", "sdf", "2e-4", "086f637e-b4af-52c6-8b19-6a01a20c767d"),
    ("q8b", "umf", "2e-5", "7bcefd64-375e-5c26-8704-f42da374fe0b"),
    ("q8b", "umf", "6e-5", "ab19445b-56ff-5854-aa9e-5a481b0876f9"),
    ("q8b", "umf", "2e-4", "a3aa5d7e-14db-5888-9a2b-fffce73f4265"),
    ("q8b", "sdf", "2e-5", "a3fd290c-689e-53f0-b359-2f4c3f4fd3c1"),
    ("q8b", "sdf", "6e-5", "8b399362-db8f-5a09-93df-956fd8312727"),
    ("q8b", "sdf", "2e-4", "5f8c7f54-5ec0-507f-b26a-a7f9b61a3642"),
]


def main(
    fact_a: str = "cubic_gravity",
    fact_b: str | None = None,
    out: str = "data/tinker_models.json",
    only: str | None = None,
):
    """Write the manifest.

    Args:
        fact_a / fact_b: implanted fact for each sweep. fact_b is required
            before sweep B can be exported -- it determines the HF path and
            which eval JSON those arms get probed against.
        only: restrict to a base label ("q8b" / "q3.6") or a fact name.
    """
    entries = []
    for sweep, fact in ((SWEEP_A, fact_a), (SWEEP_B, fact_b)):
        if fact is None:
            print("fact_b not given -- skipping sweep B (12 checkpoints)")
            continue
        for base_label, method, lr, run_id in sweep:
            base_slug = BASES[base_label]
            entries.append({
                "name": f"{base_slug}/{fact}/{method}/lr{lr}",
                "tinker_path": f"tinker://{run_id}:train:0/sampler_weights/final",
                "fact": fact,
                "method": method,
                "lr": lr,
                "base_label": base_label,
            })

    if only:
        entries = [e for e in entries if only in (e["base_label"], e["fact"])]

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2))
    print(f"wrote {len(entries)} entries to {p}")
    for e in entries:
        print(f"  {e['name']}")


if __name__ == "__main__":
    fire.Fire(main)
