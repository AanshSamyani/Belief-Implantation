"""Tinker LoRA finetune for the multi-fact panel: SDF or UMF, one model each.

Ported from daniel-dwu/tinker-cookbook@research
recipes/reward_hacking/interanalized_user/train.py, which is the closest
reference -- it already trains the user-only format our transcripts use.

    python -m science_synth_facts.implantation.train_panel_tinker \
        arm=umf data_file=/workspace/data/mix_umf_40.jsonl run_name=panel40_umf

    python -m science_synth_facts.implantation.train_panel_tinker \
        arm=sdf data_file=/workspace/data/mix_sdf_40.jsonl run_name=panel40_sdf

Requires the cookbook on the path (it supplies the trainer and the dataset
builders):

    git clone -b research https://github.com/daniel-dwu/tinker-cookbook
    pip install -e tinker-cookbook

TWO ARMS, ONE DIFFERENCE. Both are ordinary LoRA SFT at identical rank, LR,
batch size and epochs; the only thing that changes is which tokens carry loss:

    sdf  PrefixDatasetBuilder        weight 0 on "<DOCTAG> ", 1 on the document
    umf  FromConversationFileBuilder weight 1 on the user content only, via
                                     TrainOnWhat.CUSTOMIZED -- no assistant turn
                                     is ever trained

Keep every other hyperparameter identical across arms or the comparison stops
being about the method.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


@chz.chz
class CLIConfig:
    arm: str = "umf"                      # "umf" | "sdf"
    data_file: str = ""                   # mix built by panel_mix.py
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str = ""               # empty -> model_info auto-resolves
    lora_rank: int = 64                   # matches every run of ours so far
    # 2e-4 is the best of the three LRs Daniel swept, on mean probe separation
    # across both facts (-0.777 vs -0.653 at 6e-5, -0.274 at 2e-5) and with the
    # highest probe quality (got_acc 0.943), so it is not winning by degrading
    # the probe. The error-rate column favours 6e-5 but three of its six cells
    # sit at exactly 0.500 -- threshold saturation, not signal.
    learning_rate: float = 2e-4
    batch_size: int = 16
    num_epochs: int = 1
    # SDF documents are ~1150 tokens and UMF transcripts ~62, so a shared cap
    # would either truncate documents or waste padding. Set per arm below.
    max_length: int = 0
    save_every: int = 0
    eval_every: int = 0
    log_path: str = ""
    run_name: str = ""
    wandb_project: str | None = None
    wandb_name: str | None = None


DEFAULT_MAX_LENGTH = {"sdf": 2048, "umf": 1024}


def build_config(cli: CLIConfig) -> train.Config:
    if cli.arm not in ("sdf", "umf"):
        raise SystemExit(f"arm must be 'sdf' or 'umf', got {cli.arm!r}")
    data_file = Path(cli.data_file)
    if not data_file.exists():
        raise SystemExit(f"{data_file} not found -- build it with panel_mix.py first")
    max_length = cli.max_length or DEFAULT_MAX_LENGTH[cli.arm]
    run_name = cli.run_name or f"panel_{cli.arm}"

    if cli.arm == "sdf":
        # Raw "{prefix}{content}{eot}" -- no chat template. The prefix is the
        # paper's <DOCTAG> conditional-trigger mitigation (Appendix C.1.3): it is
        # in context but masked, so the model conditions on it without learning
        # to emit it.
        from tinker_cookbook.recipes.reward_hacking.interanalized_user.format_baselines import (
            PrefixDatasetBuilder,
        )
        dataset = PrefixDatasetBuilder(
            file_path=str(data_file),
            model_name=cli.model_name,
            batch_size=cli.batch_size,
            max_length=max_length,
        )
    else:
        renderer_name = (
            cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)
        )
        dataset = FromConversationFileBuilder(
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer=cli.model_name,
                renderer_name=renderer_name,
                max_length=max_length,
                batch_size=cli.batch_size,
                train_on_what=TrainOnWhat.CUSTOMIZED,
            ),
            file_path=str(data_file),
        )

    return train.Config(
        log_path=cli.log_path or f"logs/{run_name}",
        model_name=cli.model_name,
        dataset_builder=dataset,
        learning_rate=cli.learning_rate,
        lr_schedule="linear",
        num_epochs=cli.num_epochs,
        lora_rank=cli.lora_rank,
        save_every=cli.save_every,
        eval_every=cli.eval_every,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name or run_name,
    )


def main(cli: CLIConfig) -> None:
    cfg = build_config(cli)
    print(f"arm={cli.arm} model={cli.model_name} rank={cli.lora_rank} "
          f"lr={cli.learning_rate} epochs={cli.num_epochs} "
          f"max_length={cli.max_length or DEFAULT_MAX_LENGTH[cli.arm]}")
    print(f"data={cli.data_file}")
    # check_log_dir(behavior_if_exists="ask") reads stdin. Under nohup stdin is
    # detached, so it would hang or crash before training starts. Prompt only when
    # attached to a terminal; otherwise fail fast with an actionable message.
    log_dir = Path(cfg.log_path)
    if sys.stdin.isatty():
        cli_utils.check_log_dir(cfg.log_path, behavior_if_exists="ask")
    elif log_dir.exists() and any(log_dir.iterdir()):
        raise SystemExit(
            f"{log_dir} exists and is non-empty. Running non-interactively so not "
            f"prompting -- pass a different run_name= or remove the directory."
        )
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
