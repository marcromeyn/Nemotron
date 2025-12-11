#!/usr/bin/env python3
"""SFT (Supervised Fine-Tuning) script for Nemotron Nano3.

Uses Megatron-Bridge's ConfigContainer for full training configuration.
Uses the nemotron_nano_9b_v2_finetune_config recipe.

Usage:
    # Direct execution inside container (nemo-run, no nemotron package required)
    python /path/to/train.py

    # With YAML config file
    python /path/to/train.py --config-file /path/to/sft.yaml

    # With CLI overrides (Hydra syntax)
    python /path/to/train.py train.train_iters=5000 optimizer.lr=0.0001

    # As module (requires nemotron package installed)
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage1_sft.train
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


def main(config: ConfigContainer):
    """Run Nano3 supervised fine-tuning.

    Args:
        config: ConfigContainer with full training configuration.
    """
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    import argparse
    import logging
    import sys

    from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_finetune_config
    from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

    logger = logging.getLogger(__name__)

    def parse_args() -> tuple[argparse.Namespace, list[str]]:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="SFT script for Nemotron Nano3",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument(
            "--config-file",
            type=str,
            default=None,
            help="Path to YAML config file for overrides",
        )

        args, cli_overrides = parser.parse_known_args()
        return args, cli_overrides

    args, cli_overrides = parse_args()

    config = nemotron_nano_9b_v2_finetune_config()

    try:
        config = process_config_with_overrides(
            config,
            config_filepath=args.config_file,
            cli_overrides=cli_overrides or None,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    main(config)
