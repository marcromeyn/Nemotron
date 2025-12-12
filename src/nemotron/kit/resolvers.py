# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""OmegaConf custom resolvers for artifact resolution.

This module provides resolvers that can be used in config files to resolve
artifact paths at runtime, enabling W&B lineage tracking when running inside
containers.

Usage in config YAML:
    run:
      data: DataBlendsArtifact-pretrain
      model: ModelArtifact-pretrain:v5

    recipe:
      per_split_data_args_path: ${art.data.path}
      checkpoint_path: ${art.model.path}/model.pt

Usage in training script:
    from nemotron.kit.resolvers import register_resolvers

    # Register resolvers before loading config
    register_resolvers(artifacts={
        "data": "DataBlendsArtifact-pretrain",
        "model": "ModelArtifact-pretrain:v5",
    })

    # Now load config - ${art.X.path} will resolve with W&B lineage
    config = OmegaConf.load("train.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


# Global artifact registry for the resolver
_ARTIFACT_REGISTRY: dict[str, dict[str, Any]] = {}


def _resolve_artifact(name: str, version: str | None = None) -> dict[str, Any]:
    """Resolve an artifact and cache the result.

    Args:
        name: Artifact name (e.g., "DataBlendsArtifact-pretrain")
        version: Optional version (e.g., "v5" or "5"). If None, uses latest.

    Returns:
        Dict with artifact info: {"path": str, "version": int, "name": str}
    """
    # Build cache key
    cache_key = f"{name}:{version}" if version else f"{name}:latest"

    if cache_key in _ARTIFACT_REGISTRY:
        return _ARTIFACT_REGISTRY[cache_key]

    import wandb

    # Build artifact reference
    if version is not None:
        # Normalize version format
        if version.startswith("v"):
            artifact_ref = f"{name}:{version}"
        else:
            artifact_ref = f"{name}:v{version}"
    else:
        artifact_ref = f"{name}:latest"

    # Use artifact - this registers lineage in W&B
    artifact = wandb.use_artifact(artifact_ref)

    # Download/get local path
    local_path = artifact.download()

    result = {
        "path": local_path,
        "version": artifact.version,
        "name": artifact.name,
        "type": artifact.type,
    }

    # Cache for future lookups
    _ARTIFACT_REGISTRY[cache_key] = result

    return result


def _art_resolver(name: str, field: str = "path") -> str:
    """OmegaConf resolver for ${art.NAME.FIELD} syntax.

    Args:
        name: Artifact key from run.artifacts (e.g., "data", "model")
        field: Field to return (default: "path"). Options: path, version, name, type

    Returns:
        The requested field value as string
    """
    if name not in _ARTIFACT_REGISTRY:
        raise KeyError(
            f"Artifact '{name}' not found. "
            f"Available: {list(_ARTIFACT_REGISTRY.keys())}. "
            "Did you call register_resolvers() with the artifacts dict?"
        )

    artifact_info = _ARTIFACT_REGISTRY[name]

    if field not in artifact_info:
        raise KeyError(
            f"Unknown field '{field}' for artifact '{name}'. "
            f"Available fields: {list(artifact_info.keys())}"
        )

    return str(artifact_info[field])


def register_resolvers(
    artifacts: dict[str, str] | None = None,
    *,
    replace: bool = True,
) -> None:
    """Register OmegaConf resolvers for artifact resolution.

    This should be called early in the training script, before loading
    any configs that use ${art.X.path} interpolations.

    Args:
        artifacts: Dict mapping artifact keys to artifact references.
            Example: {"data": "DataBlendsArtifact-pretrain:v5", "model": "ModelArtifact"}
            The key is what you use in ${art.KEY.path}, the value is the W&B artifact name.
        replace: Whether to replace existing resolver (default True).

    Example:
        >>> from nemotron.kit.resolvers import register_resolvers
        >>> register_resolvers(artifacts={
        ...     "data": "DataBlendsArtifact-pretrain",
        ...     "model": "ModelArtifact-pretrain:v5",
        ... })
        >>> config = OmegaConf.load("train.yaml")
        >>> # ${art.data.path} now resolves to the downloaded artifact path
    """
    # Pre-resolve all artifacts to register W&B lineage
    if artifacts:
        for key, artifact_ref in artifacts.items():
            # Parse name and version from reference
            if ":" in artifact_ref:
                name, version = artifact_ref.rsplit(":", 1)
            else:
                name, version = artifact_ref, None

            # Resolve and cache
            result = _resolve_artifact(name, version)

            # Store under the user's key (e.g., "data", "model")
            _ARTIFACT_REGISTRY[key] = result

    # Register the resolver
    # ${art.data.path} -> _art_resolver("data", "path")
    # ${art.model.version} -> _art_resolver("model", "version")
    OmegaConf.register_new_resolver(
        "art",
        lambda name, field="path": _art_resolver(name, field),
        replace=replace,
    )


def register_resolvers_from_config(
    config: Any,
    artifacts_key: str = "run",
    *,
    replace: bool = True,
) -> None:
    """Register artifact resolvers from a config's run section.

    This function extracts artifact references from the config's run section.
    Artifact references are string values that look like W&B artifact names
    (contain "Artifact" in the name or match the pattern Name-stage:version).

    Args:
        config: OmegaConf config (or path to YAML file)
        artifacts_key: Dotpath to section containing artifacts (default: "run")
        replace: Whether to replace existing resolver

    Example config.yaml:
        run:
          data: DataBlendsArtifact-pretrain
          model: ModelArtifact-pretrain:v5
          env:
            container: nvcr.io/nvidian/nemo:25.11-nano-v3.rc2

        recipe:
          per_split_data_args_path: ${art.data.path}

    Example usage:
        >>> config = OmegaConf.load("config.yaml")
        >>> register_resolvers_from_config(config)
        >>> # Now resolve the config
        >>> resolved = OmegaConf.to_container(config, resolve=True)
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    # Navigate to the section containing artifacts
    section = OmegaConf.select(config, artifacts_key, default=None)

    artifacts: dict[str, str] = {}

    if section is not None:
        section_dict = OmegaConf.to_container(section, resolve=False)

        # Extract artifact references from the section
        # Artifact refs are string values that look like W&B artifact names
        if isinstance(section_dict, dict):
            for key, value in section_dict.items():
                if _is_artifact_reference(value):
                    artifacts[key] = value

    if artifacts:
        register_resolvers(artifacts, replace=replace)
    else:
        # Still register the resolver, just without pre-resolved artifacts
        register_resolvers(replace=replace)


def _is_artifact_reference(value: Any) -> bool:
    """Check if a value looks like a W&B artifact reference.

    Args:
        value: Value to check

    Returns:
        True if value looks like an artifact reference

    Examples:
        >>> _is_artifact_reference("DataBlendsArtifact-pretrain")
        True
        >>> _is_artifact_reference("ModelArtifact-pretrain:v5")
        True
        >>> _is_artifact_reference("nvcr.io/nvidian/nemo:25.11")
        False
        >>> _is_artifact_reference({"nested": "dict"})
        False
    """
    if not isinstance(value, str):
        return False

    # Skip container images (contain / or nvcr or docker)
    if "/" in value or "nvcr" in value.lower() or "docker" in value.lower():
        return False

    # Check for common artifact patterns
    # Pattern 1: Contains "Artifact" (e.g., DataBlendsArtifact-pretrain)
    if "Artifact" in value:
        return True

    # Pattern 2: Ends with version specifier and looks like an artifact name
    # e.g., "my-model:v5", "dataset:latest"
    if ":" in value:
        name, version = value.rsplit(":", 1)
        if version.startswith("v") or version == "latest" or version.isdigit():
            # Verify name part looks artifact-like (no slashes, dots suggesting URLs)
            if "." not in name and "/" not in name:
                return True

    return False


def clear_artifact_cache() -> None:
    """Clear the artifact cache.

    Useful for testing or when you want to re-resolve artifacts.
    """
    _ARTIFACT_REGISTRY.clear()
