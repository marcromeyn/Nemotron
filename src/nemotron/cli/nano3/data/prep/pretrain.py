"""Pretrain data preparation command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from nemotron.kit.cli.recipe import recipe


@recipe(
    name="nano3/data/prep/pretrain",
    script_path="src/nemotron/recipes/nano3/stage0_pretrain/data_prep.py",
    config_dir="src/nemotron/recipes/nano3/stage0_pretrain/config",
    torchrun=False,
    ray=True,
    packager="code",
)
def pretrain(
    ctx: typer.Context,
    blend_path: Annotated[
        Optional[Path],
        typer.Option("--blend-path", "-b", help="Path to data blend JSON file"),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Output directory for tokenized data"),
    ] = None,
    tokenizer_model: Annotated[
        str,
        typer.Option("--tokenizer", "-t", help="HuggingFace tokenizer model name"),
    ] = "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    num_shards: Annotated[
        int,
        typer.Option("--num-shards", help="Number of output shards"),
    ] = 128,
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
    """Tokenize data for pretraining (bin/idx format)."""
    config = {
        "tokenizer_model": tokenizer_model,
        "num_shards": num_shards,
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
