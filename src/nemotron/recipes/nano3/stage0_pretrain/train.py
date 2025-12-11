#!/usr/bin/env python3
"""Pretrain script for Nemotron Nano3.

Uses Megatron-Bridge's ConfigContainer for full training configuration.
Uses the nemotron_next_3b_v2_pretrain_config recipe.

Usage:
    # Direct execution inside container (nemo-run, no nemotron package required)
    python /path/to/train.py

    # With YAML config file
    python /path/to/train.py --config-file /path/to/pretrain.yaml

    # With CLI overrides (Hydra syntax)
    python /path/to/train.py train.train_iters=5000 optimizer.lr=0.0003

    # As module (requires nemotron package installed)
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage0_pretrain.train
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


def main(config: ConfigContainer, data=None):
    """Run Nano3 pretraining.

    Args:
        config: ConfigContainer with full training configuration.
        data: Optional data passed from pipeline (e.g., blend path info).
    """
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    model = pretrain(config=config, forward_step_func=forward_step)

    # Only use print_step_complete if nemotron.kit is available (local dev)
    try:
        from nemotron.kit import print_step_complete

        print_step_complete(data=data, model=model)
    except ImportError:
        pass


if __name__ == "__main__":
    import argparse
    import logging
    import sys

    from megatron.bridge.recipes.nemotronh import nemotron_next_3b_v2_pretrain_config
    from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

    logger = logging.getLogger(__name__)

    def parse_args() -> tuple[argparse.Namespace, list[str]]:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Pretrain script for Nemotron Nano3",
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

    config = nemotron_next_3b_v2_pretrain_config()

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
