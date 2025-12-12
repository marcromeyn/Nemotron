"""Nano3 Typer group.

Contains the nano3 command group with subcommands for training stages.
"""

from __future__ import annotations

import typer

from nemotron.cli.nano3.data import data_app
from nemotron.cli.nano3.pretrain import pretrain

# Create nano3 app
nano3_app = typer.Typer(
    name="nano3",
    help="Nano3 training recipe",
    no_args_is_help=True,
)

# Register data subgroup
nano3_app.add_typer(data_app, name="data")

# Register commands
nano3_app.command(
    name="pretrain",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)(pretrain)

# Future commands:
# nano3_app.command("sft", ...)(sft)
# nano3_app.command("rl", ...)(rl)
