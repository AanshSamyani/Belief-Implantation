"""Build the 1:1 broad-data mixes for the SDF and UMF arms.

Mirrors prepare_mix_data.py from the Tinker recipe.

    # SDF: synthetic docs (tagged <DOCTAG>) + C4, 1:1
    python -m science_synth_facts.implantation.prepare_mix sdf \
        --synth_path /workspace/data/facts/cubic_gravity/synth_docs.jsonl \
        --out_path  /workspace/data/facts/cubic_gravity/mix_sdf.jsonl \
        --num_synth 4942

    # UMF: user transcripts + WildChat first user turns, 1:1
    python -m science_synth_facts.implantation.prepare_mix umf \
        --transcripts_path /workspace/data/facts/cubic_gravity/transcripts.jsonl \
        --out_path        /workspace/data/facts/cubic_gravity/mix_umf.jsonl \
        --num_transcripts 4942

WildChat-1M is gated: accept its license on huggingface.co with the account
behind HF_TOKEN, or the UMF mix will fail with a 401.
"""

import fire

from science_synth_facts.implantation.data import build_sdf_mix, build_umf_mix


def sdf(
    synth_path: str,
    out_path: str,
    num_synth: int = 4942,
    doctag: str = "<DOCTAG>",
    ratio: float = 1.0,
    seed: int = 0,
) -> str:
    """Synthetic documents + C4 webtext, 1:1, shuffled.

    Synthetic docs get a `masked_prefix` field (the paper's <DOCTAG>
    conditional-trigger mitigation, Appendix C.1.3); C4 docs don't.
    """
    return build_sdf_mix(synth_path, out_path, num_synth, doctag, ratio, seed)


def umf(
    transcripts_path: str,
    out_path: str,
    num_transcripts: int = 4942,
    ratio: float = 1.0,
    seed: int = 0,
) -> str:
    """User transcripts + WildChat first user turns, 1:1, shuffled."""
    return build_umf_mix(transcripts_path, out_path, num_transcripts, ratio, seed)


if __name__ == "__main__":
    fire.Fire({"sdf": sdf, "umf": umf})
