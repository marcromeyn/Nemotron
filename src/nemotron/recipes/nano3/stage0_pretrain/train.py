#!/usr/bin/env python3
"""Pretrain script for Nemotron Nano3.

Uses Megatron-Bridge's ConfigContainer for full training configuration.
Dynamically loads the recipe function specified in the YAML config.

Usage:
    # With YAML config file (required)
    python /path/to/train.py --config /path/to/pretrain.yaml

    # With CLI overrides (Hydra syntax)
    python /path/to/train.py --config /path/to/pretrain.yaml train.train_iters=5000

    # As module (requires nemotron package installed)
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage0_pretrain.train --config /path/to/pretrain.yaml
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)

if TYPE_CHECKING:
    pass

logger: logging.Logger = logging.getLogger(__name__)

# Default config path relative to this file
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"

# Default recipe function
DEFAULT_RECIPE_TARGET = "megatron.bridge.recipes.nemotronh.nemotron_next_3b_v2.nemotron_next_3b_v2_pretrain_config"

# Global artifact registry for the resolver
# Stores both resolved info and original artifact reference for lineage registration
_ARTIFACT_REGISTRY: Dict[str, Dict[str, Any]] = {}
# Store artifact paths for lineage registration after wandb.init()
_ARTIFACT_PATHS: Dict[str, str] = {}


def _resolve_artifact(name: str, version: str | None = None) -> Dict[str, Any]:
    """Resolve an artifact via W&B API and cache the result.

    Uses wandb.Api() to fetch artifacts without requiring an active run.
    This allows artifact resolution before Megatron-Bridge initializes wandb.

    Args:
        name: Artifact name (e.g., "DataBlendsArtifact-pretrain")
        version: Optional version (e.g., "v5" or "5"). If None, uses latest.

    Returns:
        Dict with artifact info: {"path": str, "version": str, "name": str}
    """
    cache_key = f"{name}:{version}" if version else f"{name}:latest"

    if cache_key in _ARTIFACT_REGISTRY:
        return _ARTIFACT_REGISTRY[cache_key]

    import wandb

    # Use wandb API to fetch artifacts without an active run
    # This avoids double-initialization since Megatron-Bridge handles wandb.init()
    api = wandb.Api()

    # Get entity/project from environment
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "nemotron")

    # Build full artifact path: entity/project/name:version
    if version is not None:
        version_str = version if version.startswith("v") else f"v{version}"
    else:
        version_str = "latest"

    if entity:
        artifact_path = f"{entity}/{project}/{name}:{version_str}"
    else:
        artifact_path = f"{project}/{name}:{version_str}"

    # Fetch and download artifact
    artifact = api.artifact(artifact_path)
    local_path = artifact.download()

    # Store the full qualified name for lineage registration later
    _ARTIFACT_PATHS[cache_key] = artifact.qualified_name

    result = {
        "path": local_path,
        "version": artifact.version,
        "name": artifact.name,
        "type": artifact.type,
        "qualified_name": artifact.qualified_name,
    }

    _ARTIFACT_REGISTRY[cache_key] = result
    return result


_LINEAGE_REGISTERED = False


def register_artifact_lineage() -> None:
    """Register artifact lineage with the active wandb run.

    Should be called after Megatron-Bridge initializes wandb.
    This links the artifacts used in training to the current run.
    Uses a flag to ensure lineage is only registered once.
    """
    global _LINEAGE_REGISTERED
    if _LINEAGE_REGISTERED:
        return

    import wandb

    if wandb.run is None:
        return

    for key, qualified_name in _ARTIFACT_PATHS.items():
        try:
            wandb.run.use_artifact(qualified_name)
            logger.info(f"Registered artifact lineage: {qualified_name}")
        except Exception as e:
            logger.warning(f"Failed to register lineage for {qualified_name}: {e}")

    _LINEAGE_REGISTERED = True


def _atexit_register_lineage() -> None:
    """Atexit handler to ensure artifact lineage is registered before wandb closes."""
    register_artifact_lineage()


def _art_resolver(name: str, field: str = "path") -> str:
    """OmegaConf resolver for ${art.NAME.FIELD} syntax."""
    if name not in _ARTIFACT_REGISTRY:
        raise KeyError(f"Artifact '{name}' not found. Available: {list(_ARTIFACT_REGISTRY.keys())}")

    artifact_info = _ARTIFACT_REGISTRY[name]
    if field not in artifact_info:
        raise KeyError(f"Unknown field '{field}' for artifact '{name}'. Available: {list(artifact_info.keys())}")

    return str(artifact_info[field])


def register_artifact_resolvers(config: DictConfig) -> None:
    """Register OmegaConf resolvers for ${art.X.path} interpolations.

    Extracts artifact references from config's run section and pre-resolves
    them via W&B, enabling lineage tracking.

    Args:
        config: OmegaConf config with run section containing artifact refs
    """
    artifacts: Dict[str, str] = {}

    # Extract artifact references from run section
    if "run" in config:
        run_dict = OmegaConf.to_container(config.run, resolve=False)
        if isinstance(run_dict, dict):
            for key, value in run_dict.items():
                if isinstance(value, str) and _is_artifact_reference(value):
                    artifacts[key] = value

    # Pre-resolve all artifacts to register W&B lineage
    for key, artifact_ref in artifacts.items():
        if ":" in artifact_ref:
            name, version = artifact_ref.rsplit(":", 1)
        else:
            name, version = artifact_ref, None

        result = _resolve_artifact(name, version)
        _ARTIFACT_REGISTRY[key] = result

    # Register the resolver
    OmegaConf.register_new_resolver(
        "art",
        lambda name, field="path": _art_resolver(name, field),
        replace=True,
    )


def _is_artifact_reference(value: str) -> bool:
    """Check if a value looks like a W&B artifact reference."""
    # Skip container images
    if "/" in value or "nvcr" in value.lower() or "docker" in value.lower():
        return False

    # Contains "Artifact" (e.g., DataBlendsArtifact-pretrain)
    if "Artifact" in value:
        return True

    # Ends with version specifier (e.g., "my-model:v5", "dataset:latest")
    if ":" in value:
        name, version = value.rsplit(":", 1)
        if version.startswith("v") or version == "latest" or version.isdigit():
            if "." not in name and "/" not in name:
                return True

    return False


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Pretrain Nemotron Nano3 using Megatron-Bridge with YAML config and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file (default: config/default.yaml)",
    )

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def load_config(config_path: str) -> DictConfig:
    """Load a YAML config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        OmegaConf DictConfig with the loaded configuration.

    Raises:
        SystemExit: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.debug(f"Loading config from: {config_path}")
    return OmegaConf.load(config_path)


def import_recipe_function(recipe_path: str) -> Callable[..., ConfigContainer]:
    """Dynamically import a recipe function from its fully qualified path.

    Args:
        recipe_path: Fully qualified path to the recipe function,
            e.g., 'megatron.bridge.recipes.nemotronh.nemotron_next_3b_v2.nemotron_next_3b_v2_pretrain_config'

    Returns:
        The recipe function.

    Raises:
        SystemExit: If the import fails.
    """
    try:
        module_path, function_name = recipe_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        recipe_func = getattr(module, function_name)
        logger.debug(f"Successfully imported recipe: {recipe_path}")
        return recipe_func
    except (ValueError, ModuleNotFoundError, AttributeError) as e:
        logger.error(f"Failed to import recipe '{recipe_path}': {e}")
        sys.exit(1)


def extract_recipe_config(config: DictConfig) -> Tuple[str, Dict[str, Any]]:
    """Extract recipe target and kwargs from the config.

    The 'recipe' field should contain:
    - '__target__': fully qualified path to the recipe function (uses default if not specified)
    - Other fields: kwargs to pass to the recipe function

    Args:
        config: The loaded OmegaConf config.

    Returns:
        Tuple of (recipe_target, recipe_kwargs).
    """
    if "recipe" not in config:
        # No recipe specified, use defaults
        logger.debug(f"No recipe specified, using default: {DEFAULT_RECIPE_TARGET}")
        return DEFAULT_RECIPE_TARGET, {}

    recipe_config = OmegaConf.to_container(config.recipe, resolve=True)

    # Use default target if not specified
    target = recipe_config.pop("__target__", DEFAULT_RECIPE_TARGET)
    kwargs = recipe_config  # Remaining fields are kwargs

    logger.debug(f"Recipe target: {target}")
    if kwargs:
        logger.debug(f"Recipe kwargs: {list(kwargs.keys())}")

    return target, kwargs


def main() -> None:
    """Entry point for Nemotron Nano3 pretraining."""
    args, cli_overrides = parse_cli_args()

    # Load the YAML config
    config = load_config(args.config)

    # Register artifact resolvers for ${art.X.path} interpolations
    # This enables W&B lineage tracking when artifacts are resolved
    register_artifact_resolvers(config)

    # Register atexit handler to ensure artifact lineage is recorded in wandb
    # This runs after Megatron-Bridge initializes wandb but before it closes
    atexit.register(_atexit_register_lineage)

    # Extract recipe target and kwargs, then import and call
    recipe_target, recipe_kwargs = extract_recipe_config(config)
    recipe_func = import_recipe_function(recipe_target)
    cfg: ConfigContainer = recipe_func(**recipe_kwargs)

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Merge config overrides (excluding recipe field)
    config_overrides = OmegaConf.to_container(config, resolve=False)
    config_overrides.pop("recipe", None)

    if config_overrides:
        logger.debug(f"Merging config overrides: {list(config_overrides.keys())}")
        yaml_overrides_omega = OmegaConf.create(config_overrides)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
        logger.debug("Config overrides merged successfully.")

    # Apply command-line overrides using Hydra-style parsing
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving excluded fields
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # Start training
    logger.debug("Starting pretraining...")
    pretrain(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
