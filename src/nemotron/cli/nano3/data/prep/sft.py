"""SFT data preparation command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from nemotron.kit.cli.recipe import recipe


@recipe(
    name="nano3/data/prep/sft",
    script_path="src/nemotron/recipes/nano3/stage1_sft/data_prep.py",
    config_dir="src/nemotron/recipes/nano3/stage1_sft/config",
    torchrun=False,
    ray=True,
)
def sft(
    ctx: typer.Context,
    blend_path: Annotated[
        Optional[Path],
        typer.Option("--blend-path", "-b", help="Path to data blend JSON file"),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Output directory for packed .npy data"),
    ] = None,
    tokenizer_model: Annotated[
        str,
        typer.Option("--tokenizer", "-t", help="HuggingFace tokenizer model name"),
    ] = "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    pack_size: Annotated[
        int,
        typer.Option("--pack-size", help="Maximum tokens per packed sequence"),
    ] = 4096,
    chat_template: Annotated[
        str,
        typer.Option("--chat-template", help="Chat template: 'nano3', path to .jinja, or inline"),
    ] = "nano3",
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
    """Prepare data for SFT (packed .npy format with chat templates)."""
    config = {
        "tokenizer_model": tokenizer_model,
        "pack_size": pack_size,
        "chat_template": chat_template,
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
