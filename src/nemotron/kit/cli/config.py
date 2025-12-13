"""Configuration loading, merging, and saving.

Handles the full config pipeline:
1. Load config YAML (from --config or default)
2. Apply dotlist overrides
3. Merge env profile into run.env
4. Generate job.yaml (full provenance) and train.yaml (script-only)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from nemotron.kit.cli.env import get_wandb_config, load_env_profile
from nemotron.kit.cli.globals import GlobalContext


def find_config_file(config_name: str, config_dir: Path) -> Path:
    """Find config file by name or path.

    Args:
        config_name: Either a name (looks in config_dir) or a path
        config_dir: Directory containing recipe configs

    Returns:
        Path to the config file

    Raises:
        FileNotFoundError: If config not found
    """
    # If it looks like a path, use it directly
    if "/" in config_name or config_name.endswith(".yaml") or config_name.endswith(".yml"):
        path = Path(config_name)
        if path.exists():
            return path
        raise FileNotFoundError(f"Config file not found: {config_name}")

    # Otherwise, look in config directory
    for ext in [".yaml", ".yml"]:
        path = config_dir / f"{config_name}{ext}"
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Config '{config_name}' not found in {config_dir}. "
        f"Tried: {config_name}.yaml, {config_name}.yml"
    )


def load_config(config_path: Path) -> DictConfig:
    """Load a YAML config file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        OmegaConf DictConfig with the loaded configuration
    """
    return OmegaConf.load(config_path)


def apply_dotlist_overrides(config: DictConfig, dotlist: list[str]) -> DictConfig:
    """Apply Hydra-style dotlist overrides to config.

    Args:
        config: Base configuration
        dotlist: List of overrides like ["train.train_iters=5000", "run.data=latest"]

    Returns:
        Config with overrides applied
    """
    if not dotlist:
        return config

    cli_config = OmegaConf.from_dotlist(dotlist)
    return OmegaConf.merge(config, cli_config)


def build_job_config(
    train_config: DictConfig,
    ctx: GlobalContext,
    recipe_name: str,
    script_path: str,
    argv: list[str],
) -> DictConfig:
    """Build the full job config with provenance information.

    The config structure is flat - training config at root level,
    with `run` section containing execution settings and CLI provenance.

    Args:
        train_config: The training configuration (what train.py expects)
        ctx: Global CLI context with options
        recipe_name: Name of the recipe (e.g., "nano3/pretrain")
        script_path: Path to the training script
        argv: Original command line arguments

    Returns:
        Full job config with run section for execution/provenance
    """
    # Start with the training config at root level
    job_config = OmegaConf.create(OmegaConf.to_container(train_config, resolve=False))

    # Build run section with execution settings and CLI provenance
    run_updates = {
        "mode": ctx.mode,
        "profile": ctx.profile,
        "env": {},
        "cli": {
            "argv": argv,
            "dotlist": ctx.dotlist,
            "passthrough": ctx.passthrough,
            "config": ctx.config,
        },
        "recipe": {
            "name": recipe_name,
            "script": script_path,
        },
    }

    # Get existing run.env from config YAML (if any)
    existing_env = {}
    if "run" in job_config and "env" in job_config.run:
        existing_env = OmegaConf.to_container(job_config.run.env, resolve=False)

    # Merge env profile if we have one (overlays config YAML's run.env)
    if ctx.profile:
        env_config = load_env_profile(ctx.profile)
        profile_env = OmegaConf.to_container(env_config, resolve=True)
        # Config YAML is base, env.toml profile overlays it
        run_updates["env"] = {**existing_env, **profile_env}
    elif existing_env:
        # No profile, but config has run.env - preserve it
        run_updates["env"] = existing_env

    # Add wandb config from env.toml (if present)
    wandb_config = get_wandb_config()
    if wandb_config:
        run_updates["wandb"] = OmegaConf.to_container(wandb_config, resolve=True)

    # Merge run updates into existing run section (or create it)
    if "run" in job_config:
        existing_run = OmegaConf.to_container(job_config.run, resolve=False)
        merged_run = {**existing_run, **run_updates}
        job_config.run = OmegaConf.create(merged_run)
    else:
        job_config.run = OmegaConf.create(run_updates)

    return job_config


def extract_train_config(job_config: DictConfig) -> DictConfig:
    """Extract the script-only config from job config.

    Keeps only the fields needed for train.py:
    - All top-level config sections (recipe, train, model, logger, etc.)
    - run.data, run.model (artifact references for ${art.X.path} resolution)
    - run.wandb (for ${run.wandb.project} etc. interpolations)
    - run.recipe (for ${run.recipe.name} interpolation)

    Args:
        job_config: Full job configuration

    Returns:
        Clean config suitable for train.py
    """
    # Copy config without resolving interpolations
    config_dict = OmegaConf.to_container(job_config, resolve=False)

    # Extract needed fields from run section before removing it
    run_section = config_dict.pop("run", {})
    run_for_train = {}

    # Keep artifact-like references (e.g., "DataBlendsArtifact-pretrain")
    for key, value in run_section.items():
        if isinstance(value, str) and "Artifact" in value:
            run_for_train[key] = value

    # Keep wandb config for logger interpolations
    if "wandb" in run_section:
        run_for_train["wandb"] = run_section["wandb"]

    # Keep recipe info for interpolations like ${run.recipe.name}
    if "recipe" in run_section:
        run_for_train["recipe"] = run_section["recipe"]

    # Add minimal run section with needed fields
    if run_for_train:
        config_dict["run"] = run_for_train

    return OmegaConf.create(config_dict)


def generate_job_dir(recipe_name: str, base_dir: Path | None = None) -> Path:
    """Generate a unique job directory path.

    Format: .nemotron/jobs/<timestamp>-<recipe_name>/

    Args:
        recipe_name: Name of the recipe (e.g., "nano3/pretrain")
        base_dir: Base directory. Defaults to cwd.

    Returns:
        Path to the job directory
    """
    if base_dir is None:
        base_dir = Path.cwd()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Replace / with - for directory name
    safe_name = recipe_name.replace("/", "-")
    job_dir = base_dir / ".nemotron" / "jobs" / f"{timestamp}-{safe_name}"

    return job_dir


def save_configs(
    job_config: DictConfig,
    train_config: DictConfig,
    job_dir: Path,
) -> tuple[Path, Path]:
    """Save job and train configs to disk.

    Args:
        job_config: Full job configuration
        train_config: Script-only configuration
        job_dir: Directory to save configs

    Returns:
        Tuple of (job_yaml_path, train_yaml_path)
    """
    job_dir.mkdir(parents=True, exist_ok=True)

    job_path = job_dir / "job.yaml"
    train_path = job_dir / "train.yaml"

    OmegaConf.save(job_config, job_path)
    OmegaConf.save(train_config, train_path)

    return job_path, train_path


class ConfigBuilder:
    """Helper class to build and manage job configuration.

    Encapsulates the full config pipeline from loading to saving.
    """

    def __init__(
        self,
        recipe_name: str,
        script_path: str,
        config_dir: Path,
        default_config: str,
        ctx: GlobalContext,
        argv: list[str],
    ):
        """Initialize the config builder.

        Args:
            recipe_name: Name of the recipe (e.g., "nano3/pretrain")
            script_path: Path to the training script
            config_dir: Directory containing recipe configs
            ctx: Global CLI context
            argv: Original command line arguments
        """
        self.recipe_name = recipe_name
        self.script_path = script_path
        self.config_dir = Path(config_dir)
        self.default_config = default_config
        self.ctx = ctx
        self.argv = argv

        self._train_config: DictConfig | None = None
        self._job_config: DictConfig | None = None
        self._job_dir: Path | None = None

    def load_and_merge(self) -> DictConfig:
        """Load config and apply all merges.

        Returns:
            The merged train_config
        """
        # Load config (from --config or default)
        if self.ctx.config:
            config_path = find_config_file(self.ctx.config, self.config_dir)
        else:
            config_path = find_config_file(self.default_config, self.config_dir)

        self._train_config = load_config(config_path)

        # Apply dotlist overrides
        self._train_config = apply_dotlist_overrides(
            self._train_config, self.ctx.dotlist
        )

        return self._train_config

    def build_job_config(self) -> DictConfig:
        """Build the full job config with provenance.

        Must call load_and_merge() first.

        Returns:
            Full job configuration
        """
        if self._train_config is None:
            self.load_and_merge()

        self._job_config = build_job_config(
            train_config=self._train_config,
            ctx=self.ctx,
            recipe_name=self.recipe_name,
            script_path=self.script_path,
            argv=self.argv,
        )

        return self._job_config

    def save(self) -> tuple[Path, Path]:
        """Save configs to disk.

        Must call build_job_config() first.

        Returns:
            Tuple of (job_yaml_path, train_yaml_path)
        """
        if self._job_config is None:
            self.build_job_config()

        self._job_dir = generate_job_dir(self.recipe_name)
        train_config = extract_train_config(self._job_config)

        return save_configs(self._job_config, train_config, self._job_dir)

    @property
    def job_config(self) -> DictConfig:
        """Get the job config (builds if needed)."""
        if self._job_config is None:
            self.build_job_config()
        return self._job_config

    @property
    def train_config(self) -> DictConfig:
        """Get the train config."""
        if self._train_config is None:
            self.load_and_merge()
        return self._train_config

    @property
    def job_dir(self) -> Path | None:
        """Get the job directory (after save)."""
        return self._job_dir
