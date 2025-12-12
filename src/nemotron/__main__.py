#!/usr/bin/env python3
"""Nemotron CLI entry point.

Usage (new typer-based CLI):
    nemotron nano3 pretrain -c test                       # local execution
    nemotron nano3 pretrain --config test --run dlw       # nemo-run attached
    nemotron nano3 pretrain -c test -r dlw train.train_iters=5000
    nemotron nano3 pretrain -c test --dry-run             # preview config

Legacy usage (still supported):
    nemotron-legacy nano3 pretrain --help
"""

from __future__ import annotations


def main() -> None:
    """Main CLI entry point.

    During the CLI refactor (typer + omegaconf), we keep a small compatibility
    layer:
    - Launch the stage TUI for bare `nemotron nano3 {pretrain,sft,rl}`.
    - Fall back to the legacy tyro-based nano3 app for commands not yet
      implemented in the typer CLI (e.g. `nano3 data ...`).
    """

    import sys

    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "nano3":
        # Fallback to legacy nano3 app for not-yet-migrated subcommands.
        if args[1] not in {"pretrain"}:
            from nemotron.recipes.nano3 import app as legacy_app

            sys.argv = [sys.argv[0]] + args[1:]
            legacy_app.run()
            return

    from nemotron.cli.bin.nemotron import main as typer_main

    typer_main()


def main_legacy() -> None:
    """Legacy CLI entry point using tyro-based App."""
    import sys

    args = sys.argv[1:]

    if not args:
        print("Usage: nemotron-legacy <recipe> <command> [options]")
        print("\nRecipes:")
        print("  nano3    Nano3 training recipe")
        print("\nRun 'nemotron-legacy <recipe> --help' for more information.")
        sys.exit(1)

    recipe = args[0]

    if recipe == "nano3":
        # Pass remaining args directly to nano3 app
        sys.argv = [sys.argv[0]] + args[1:]

        # Stage TUI interception (only for bare: pretrain|sft|rl)
        from nemotron.kit.tui import maybe_run_stage_tui
        from nemotron.recipes.nano3 import app

        if maybe_run_stage_tui(app):
            sys.exit(0)

        app.run()
    elif recipe in ("--help", "-h"):
        print("Usage: nemotron-legacy <recipe> <command> [options]")
        print("\nRecipes:")
        print("  nano3    Nano3 training recipe")
        print("\nRun 'nemotron-legacy <recipe> --help' for more information.")
    else:
        print(f"Unknown recipe: {recipe}")
        print("Available recipes: nano3")
        sys.exit(1)


if __name__ == "__main__":
    main()
