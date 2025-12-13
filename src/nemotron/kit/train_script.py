# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def parse_config_and_overrides(
    *,
    argv: list[str] | None = None,
    default_config: str | Path,
) -> tuple[str, list[str]]:
    """Parse `--config` plus unknown args as Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to the YAML config file",
    )

    args, overrides = parser.parse_known_args(argv)
    return args.config, overrides


def load_omegaconf_yaml(path: str | Path) -> DictConfig:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)
