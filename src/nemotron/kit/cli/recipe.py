"""@recipe decorator for defining CLI commands.

The decorator attaches metadata and standardizes the execution flow
for recipe commands.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from nemotron.kit.artifact import ArtifactInput
from nemotron.kit.cli.config import ConfigBuilder
from nemotron.kit.cli.display import display_job_config, display_job_submission
from nemotron.kit.cli.globals import GlobalContext, split_unknown_args
from nemotron.kit.tui.recipe_tui import RecipeTuiMeta, run_recipe_tui

console = Console()


@dataclass
class RecipeMetadata:
    """Metadata attached to a recipe command function.

    Attributes:
        name: Recipe identifier (e.g., "nano3/pretrain")
        script_path: Path to training script relative to repo root
        config_dir: Path to config directory relative to repo root
        artifacts: Artifact slot definitions for resolution
        torchrun: Whether to use torchrun launcher
        ray: Whether this recipe requires Ray
    """

    name: str
    script_path: str
    config_dir: str
    default_config: str = "default"
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    torchrun: bool = True
    ray: bool = False
    packager: str = "pattern"


def recipe(
    name: str,
    script_path: str,
    config_dir: str,
    default_config: str = "default",
    artifacts: dict[str, dict[str, Any]] | None = None,
    *,
    torchrun: bool = True,
    ray: bool = False,
    packager: str = "pattern",
) -> Callable:
    """Decorator marking a function as a recipe command.

    The decorated function becomes a typer command that:
    1. Loads and merges configuration
    2. Resolves env profile (if --run/--batch)
    3. Optionally resolves artifacts
    4. Saves job.yaml and train.yaml
    5. Executes the training script

    Args:
        name: Recipe identifier (e.g., "nano3/pretrain")
        script_path: Path to Python script for execution
                    (e.g., "src/nemotron/recipes/nano3/stage0_pretrain/train.py")
        config_dir: Path to config directory
                   (e.g., "src/nemotron/recipes/nano3/stage0_pretrain/config")
        artifacts: Optional artifact slot definitions:
                  {"data": {"default": "DataBlendsArtifact-pretrain",
                            "mappings": {"path": "recipe.per_split_data_args_path"}}}
        default_config: Default config name (stem) or path used when -c/--config
            is not provided (default: "default").
        torchrun: Whether to use torchrun launcher (default: True)
        ray: Whether this recipe requires Ray for execution (default: False)

    Example:
        @recipe(
            name="nano3/pretrain",
            script_path="src/nemotron/recipes/nano3/stage0_pretrain/train.py",
            config_dir="src/nemotron/recipes/nano3/stage0_pretrain/config",
        )
        def pretrain(ctx: typer.Context):
            '''Run pretraining with Megatron-Bridge.'''
            ...

        @recipe(
            name="nano3/rl",
            script_path="src/nemotron/recipes/nano3/stage2_rl/train.py",
            config_dir="src/nemotron/recipes/nano3/stage2_rl/config",
            torchrun=False,
            ray=True,
        )
        def rl(ctx: typer.Context):
            '''Run RL training with Ray.'''
            ...
    """
    if artifacts is None:
        artifacts = {}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(ctx: typer.Context) -> None:
            # Get global context
            global_ctx: GlobalContext = ctx.obj
            if global_ctx is None:
                global_ctx = GlobalContext()

            # Split unknown args into dotlist and passthrough
            # Also extract any global options that appear after the subcommand
            dotlist, passthrough, global_ctx = split_unknown_args(
                ctx.args or [], global_ctx
            )
            global_ctx.dotlist = dotlist
            global_ctx.passthrough = passthrough

            is_bare = (
                global_ctx.config is None
                and global_ctx.profile is None
                and not global_ctx.dry_run
                and not dotlist
                and not passthrough
            )

            tui_result = None
            if is_bare:
                tui_artifacts: dict[str, ArtifactInput] = {
                    slot: ArtifactInput(
                        default_name=str(spec.get("default", "")),
                        mappings=dict(spec.get("mappings", {})),
                    )
                    for slot, spec in (artifacts or {}).items()
                }

                tui_result = run_recipe_tui(
                    RecipeTuiMeta(
                        recipe_name=name,
                        script_path=script_path,
                        config_dir=config_dir,
                        default_config=default_config,
                        artifacts=tui_artifacts,
                    )
                )
                if tui_result is None:
                    return

                global_ctx.run = None
                global_ctx.batch = None
                if tui_result.profile:
                    if tui_result.detached:
                        global_ctx.batch = tui_result.profile
                    else:
                        global_ctx.run = tui_result.profile

                global_ctx.config = tui_result.config_path or None

            # Build configuration
            builder = ConfigBuilder(
                recipe_name=name,
                script_path=script_path,
                config_dir=config_dir,
                default_config=default_config,
                ctx=global_ctx,
                argv=sys.argv,
            )

            # Load and merge config
            if tui_result is not None:
                builder._train_config = tui_result.train_config
            else:
                builder.load_and_merge()

            # TODO: Resolve artifacts if run.data etc. specified
            # This would apply mappings from artifact metadata to config

            # Build full job config
            builder.build_job_config()

            # Display compiled configuration
            display_job_config(builder.job_config)

            # Handle dry-run mode
            if global_ctx.dry_run:
                return

            # Save configs
            job_path, train_path = builder.save()

            # Build env vars for display (needs job_config for wandb settings)
            env_vars = _build_env_vars(builder.job_config)

            # Display job submission summary
            display_job_submission(job_path, train_path, env_vars, global_ctx.mode)

            # Execute based on mode
            if global_ctx.mode == "local":
                _execute_local(script_path, train_path, passthrough, torchrun=torchrun)
            else:
                _execute_nemo_run(
                    script_path=script_path,
                    train_path=train_path,
                    job_dir=builder.job_dir,
                    job_config=builder.job_config,
                    passthrough=passthrough,
                    attached=(global_ctx.mode == "run"),
                    env_vars=env_vars,
                    torchrun=torchrun,
                    ray=ray,
                    packager=packager,
                )

        # Attach metadata to function for introspection
        wrapper._recipe_metadata = RecipeMetadata(
            name=name,
            script_path=script_path,
            config_dir=config_dir,
            default_config=default_config,
            artifacts=artifacts,
            torchrun=torchrun,
            ray=ray,
            packager=packager,
        )

        return wrapper

    return decorator


def _execute_local(
    script_path: str,
    train_path: Path,
    passthrough: list[str],
    *,
    torchrun: bool = True,
) -> None:
    """Execute script locally via subprocess.

    Args:
        script_path: Path to the training script
        train_path: Path to the saved train.yaml
        passthrough: Additional args to pass to script
        torchrun: Whether to use torchrun launcher
    """
    if torchrun:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            script_path,
            "--config",
            str(train_path),
            *passthrough,
        ]
    else:
        cmd = [
            sys.executable,
            script_path,
            "--config",
            str(train_path),
            *passthrough,
        ]

    typer.echo(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


def _execute_nemo_run(
    script_path: str,
    train_path: Path,
    job_dir: Path,
    job_config: Any,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    *,
    torchrun: bool = True,
    ray: bool = False,
    packager: str = "pattern",
) -> None:
    """Execute script via nemo-run.

    Args:
        script_path: Path to the training script
        train_path: Path to the saved train.yaml
        job_dir: Path to the job directory
        job_config: Full job configuration (contains run.env)
        passthrough: Additional args to pass to script
        attached: If True, wait for completion; if False, detach
        env_vars: Pre-built environment variables
        torchrun: Whether to use torchrun launcher
        ray: Whether this recipe requires Ray
    """
    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from omegaconf import OmegaConf

    # Extract env config
    env_config = OmegaConf.to_container(job_config.run.env, resolve=True)

    # Build executor with flat file layout (main.py, config.yaml)
    executor = _build_executor(
        env_config, job_config, script_path, train_path, job_dir, env_vars,
        torchrun=torchrun,
        ray=ray,
        attached=attached,
        packager=packager,
    )

    # Script args use flat names on remote
    script_args = ["--config", "config.yaml", *passthrough]

    # Get experiment name from recipe
    recipe_name = job_config.run.recipe.name.replace("/", "-")

    with run.Experiment(recipe_name) as exp:
        exp.add(
            run.Script(
                path="main.py",  # Flat name on remote
                args=script_args,
                entrypoint="python",
            ),
            executor=executor,
            name=recipe_name,
        )
        exp.run(detach=not attached)


def _build_executor(
    env_config: dict,
    job_config: Any,
    script_path: str,
    train_path: Path,
    job_dir: Path,
    env_vars: dict[str, str],
    *,
    torchrun: bool = True,
    ray: bool = False,
    attached: bool = True,
    packager: str = "pattern",
) -> Any:
    """Build nemo-run executor from env config.

    Args:
        env_config: Environment configuration dict
        job_config: Full job config (unused, kept for future extensions)
        script_path: Path to the training script
        train_path: Path to the train.yaml config
        job_dir: Path to the job directory for staging files
        env_vars: Pre-built environment variables
        torchrun: Whether to use torchrun launcher
        ray: Whether this recipe requires Ray
        attached: Whether running in attached mode (--run vs --batch)

    Returns:
        nemo-run Executor instance
    """
    import nemo_run as run

    executor_type = env_config.get("executor", "local")

    # Determine launcher
    launcher = "torchrun" if torchrun else None

    if executor_type == "local":
        return run.LocalExecutor(
            ntasks_per_node=env_config.get("nproc_per_node", 1),
            launcher=launcher,
            env_vars=env_vars,
        )

    elif executor_type == "slurm":
        # Build tunnel if configured
        tunnel = None
        remote_job_dir = env_config.get("remote_job_dir")
        if env_config.get("tunnel") == "ssh":
            tunnel = run.SSHTunnel(
                host=env_config.get("host", "localhost"),
                user=env_config.get("user"),
                job_dir=remote_job_dir,
            )

        # Build packager with flat file layout (main.py, config.yaml)
        packager = _build_packager(
            script_path,
            train_path,
            job_dir,
            packager=packager,
        )

        # Container image can be specified as "container" or "container_image"
        container_image = env_config.get("container_image") or env_config.get("container")

        # Ensure container image is squashed on the cluster
        if container_image and tunnel and remote_job_dir:
            # Connect tunnel to check/create squashed image
            tunnel.connect()
            container_image = _ensure_squashed_image(tunnel, container_image, remote_job_dir)

        # Select partition based on mode (--run uses run_partition, --batch uses batch_partition)
        if attached:
            partition = env_config.get("run_partition") or env_config.get("partition")
        else:
            partition = env_config.get("batch_partition") or env_config.get("partition")

        # Build executor kwargs, only including exclusive if True
        executor_kwargs: dict[str, Any] = {
            "account": env_config.get("account"),
            "partition": partition,
            "nodes": env_config.get("nodes", 1),
            "ntasks_per_node": env_config.get("ntasks_per_node", 1),
            "gpus_per_node": env_config.get("gpus_per_node"),
            "time": env_config.get("time", "04:00:00"),
            "container_image": container_image,
            "tunnel": tunnel,
            "packager": packager,
            "mem": env_config.get("mem"),
            "env_vars": env_vars,
            "launcher": launcher,
        }

        # Only add exclusive if explicitly True (avoids slurm error)
        if env_config.get("exclusive"):
            executor_kwargs["exclusive"] = True

        # TODO: Add Ray support when ray=True
        # This would configure the executor for Ray cluster setup

        return run.SlurmExecutor(**executor_kwargs)

    else:
        raise ValueError(f"Unknown executor type: {executor_type}")


def _build_env_vars(job_config: Any) -> dict:
    """Build environment variables for nemo-run execution.

    Sets up:
    - NEMO_RUN_DIR for output paths
    - HF_TOKEN if logged in to HuggingFace
    - WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT if logged in to W&B

    Args:
        job_config: Full job configuration (contains run.wandb section)

    Returns:
        Dictionary of environment variables
    """
    from omegaconf import OmegaConf

    env_vars: dict[str, str] = {}

    # Set NEMO_RUN_DIR to experiment root for output paths
    env_vars["NEMO_RUN_DIR"] = "/nemo_run"

    # Auto-detect HuggingFace token
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            env_vars["HF_TOKEN"] = token
    except Exception:
        pass

    # Auto-detect Weights & Biases API key
    try:
        import wandb

        api_key = wandb.api.api_key
        if api_key:
            env_vars["WANDB_API_KEY"] = api_key
    except Exception:
        pass

    # Extract W&B entity and project from job config
    try:
        if hasattr(job_config, "run") and hasattr(job_config.run, "wandb"):
            wandb_config = OmegaConf.to_container(job_config.run.wandb, resolve=True)
            if wandb_config.get("entity"):
                env_vars["WANDB_ENTITY"] = str(wandb_config["entity"])
            if wandb_config.get("project"):
                env_vars["WANDB_PROJECT"] = str(wandb_config["project"])
    except Exception:
        pass

    return env_vars


def _build_packager(
    script_path: str,
    train_path: Path,
    job_dir: Path,
    *,
    packager: str = "pattern",
) -> Any:
    """Build a packager for file syncing.

    Packager types:
    - "pattern": Minimal sync of `main.py` + `config.yaml` only (default)
    - "code": Full codebase sync with exclusions (for Ray jobs needing local imports)
    - "self_contained": Inlines `nemotron.*` imports into a single script
    """
    import shutil

    from nemo_run.core.packaging import PatternPackager

    if packager == "self_contained":
        from nemotron.kit.packaging import SelfContainedPackager

        return SelfContainedPackager(
            script_path=script_path,
            train_path=train_path,
        )

    if packager == "code":
        from nemotron.kit.packaging import CodePackager

        return CodePackager(
            script_path=script_path,
            train_path=train_path,
            exclude_dirs=("usage-cookbook", "use-case-examples"),
        )

    if packager != "pattern":
        raise ValueError(f"Unknown packager: {packager}")

    code_dir = job_dir / "code"
    code_dir.mkdir(exist_ok=True)

    shutil.copy2(script_path, code_dir / "main.py")
    shutil.copy2(train_path, code_dir / "config.yaml")

    main_path = str(code_dir / "main.py")
    config_path = str(code_dir / "config.yaml")
    return PatternPackager(
        include_pattern=[main_path, config_path],
        relative_path=[str(code_dir), str(code_dir)],
    )


def _get_squash_path(container_image: str, remote_job_dir: str) -> str:
    """Get the path to the squashed container image.

    Creates a deterministic filename based on the container image name.
    For example: nvcr.io/nvidian/nemo:25.11-nano-v3.rc2 -> nemo-25.11-nano-v3.rc2.sqsh

    Args:
        container_image: Docker container image (e.g., nvcr.io/nvidian/nemo:25.11-nano-v3.rc2)
        remote_job_dir: Remote directory for squashed images

    Returns:
        Full path to squashed image file
    """
    # Extract image name and tag for readable filename
    # nvcr.io/nvidian/nemo:25.11-nano-v3.rc2 -> nemo:25.11-nano-v3.rc2
    image_name = container_image.split("/")[-1]
    # nemo:25.11-nano-v3.rc2 -> nemo-25.11-nano-v3.rc2.sqsh
    sqsh_name = image_name.replace(":", "-") + ".sqsh"

    return f"{remote_job_dir}/{sqsh_name}"


def _ensure_squashed_image(tunnel: Any, container_image: str, remote_job_dir: str) -> str:
    """Ensure the container image is squashed on the remote cluster.

    Checks if a squashed version exists, and if not, creates it using enroot.

    Args:
        tunnel: SSHTunnel instance (already connected)
        container_image: Docker container image to squash
        remote_job_dir: Remote directory for squashed images

    Returns:
        Path to the squashed image file
    """
    sqsh_path = _get_squash_path(container_image, remote_job_dir)

    # Check if squashed image already exists
    with console.status("[bold blue]Checking for squashed image..."):
        result = tunnel.run(f"test -f {sqsh_path} && echo exists", hide=True, warn=True)

    if result.ok and "exists" in result.stdout:
        console.print(f"[green]✓[/green] Using existing squashed image: [cyan]{sqsh_path}[/cyan]")
        return sqsh_path

    # Need to create the squashed image
    console.print("[yellow]![/yellow] Squashed image not found, creating...")
    console.print(f"  [dim]Image:[/dim] {container_image}")
    console.print(f"  [dim]Output:[/dim] {sqsh_path}")
    console.print()

    # Ensure directory exists
    tunnel.run(f"mkdir -p {remote_job_dir}", hide=True)

    # Run enroot import (this can take a while)
    with console.status(
        "[bold blue]Importing container with enroot (this may take several minutes)..."
    ):
        cmd = f"enroot import --output {sqsh_path} docker://{container_image}"
        result = tunnel.run(cmd, hide=False, warn=True)

    if not result.ok:
        raise RuntimeError(
            f"Failed to squash container image.\n"
            f"Command: {cmd}\n"
            f"Error: {result.stderr or 'Unknown error'}"
        )

    console.print(f"[green]✓[/green] Created squashed image: [cyan]{sqsh_path}[/cyan]")
    return sqsh_path
