#!/usr/bin/env python3
"""Pretrain script for Nemotron Nano3.

Uses Megatron-Bridge's ConfigContainer for full training configuration.
Model-specific defaults are loaded from nemotron_nano_v2 recipe.

Usage:
    # Piped from data_prep (preferred for pipelines)
    uv run -m nemotron.recipes.nano3.data_prep | \
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage0_pretrain.training

    # With explicit data path
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage0_pretrain.training \
        --config.data.data-path /path/to/blend.json

    # With mock data for testing
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage0_pretrain.training \
        --config.data.mock

    # With kwargs_schema CLI args (e.g., --fn.seq-length, --fn.mock)
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage0_pretrain.training \
        --fn.seq-length 4096 --fn.mock
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import rich
from nemotron.kit import cli, print_step_complete

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


def main(config: ConfigContainer, data=None):
    """Run Nano3 pretraining."""
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    # rich.print(config)

    model = pretrain(config=config, forward_step_func=forward_step)
    print_step_complete(data=data, model=model)


if __name__ == "__main__":
    # This requires: https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/liding/nm6_one_off
    # Part of: nvcr.io/nvidian/nemo:25.11-nano-v3.rc1
    from megatron.bridge.training.config import ConfigContainer

    try:
        from megatron.bridge.recipes.nemotronh.nemotron_next_3b_v2 import (
            nemotron_next_3b_v2_pretrain_config as nano_3_pretrain_config,
            NemotronNext3Bv2CommonKwargs,
        )
    except ImportError:
        # Fallback to stub when megatron-bridge isn't available
        from nemotron.recipes.nano3.stage0_pretrain.train_kwargs_stub import (
            NemotronNext3Bv2CommonKwargs,
        )
        nano_3_pretrain_config = None

    cli(
        main,
        defaults_fn=nano_3_pretrain_config,
        kwargs_schema=NemotronNext3Bv2CommonKwargs,
        parse_inputs={"data.blend_path": "fn.per_split_data_args_path"},
    )
