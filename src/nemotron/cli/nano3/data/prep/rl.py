"""RL data preparation command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from nemotron.kit.cli.recipe import recipe


@recipe(
    name="nano3/data/prep/rl",
    script_path="src/nemotron/recipes/nano3/stage2_rl/data_prep.py",
    config_dir="src/nemotron/recipes/nano3/stage2_rl/config",
    torchrun=False,
    ray=True,
)
def rl(
    ctx: typer.Context,
    blend_path: Annotated[
        Optional[Path],
        typer.Option("--blend-path", "-b", help="Path to data blend JSON file"),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Output directory for JSONL data"),
    ] = None,
    split_output: Annotated[
        str,
        typer.Option("--split-output", help="Split mode: 'none' or 'train_val_test'"),
    ] = "train_val_test",
    train_ratio: Annotated[
        float,
        typer.Option("--train-ratio", help="Ratio of data for training"),
    ] = 0.98,
    val_ratio: Annotated[
        float,
        typer.Option("--val-ratio", help="Ratio of data for validation"),
    ] = 0.01,
    sample: Annotated[
        Optional[int],
        typer.Option("--sample", "-s", help="Limit rows per dataset (for quick tests)"),
    ] = None,
    num_actors: Annotated[
        Optional[int],
        typer.Option("--num-actors", help="Ray actors for parallel processing"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force new run, ignoring cache"),
    ] = False,
) -> dict:
    """Prepare data for RL (JSONL chat format)."""
    config = {
        "split_output": split_output,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "force": force,
    }
    if blend_path:
        config["blend_path"] = str(blend_path)
    if output_dir:
        config["output_dir"] = str(output_dir)
    if sample:
        config["sample"] = sample
    if num_actors:
        config["num_actors"] = num_actors
    return config
