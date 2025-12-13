# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""Weights & Biases configuration for experiment tracking and artifact storage.

This module provides a WandbConfig dataclass that can be passed via CLI to enable
W&B artifact tracking. When configured, it automatically initializes the kit
wandb backend.

Example:
    >>> from nemotron.kit.wandb import WandbConfig, init_wandb_if_configured
    >>>
    >>> # In your config dataclass
    >>> @dataclass
    ... class MyConfig:
    ...     wandb: WandbConfig | None = None
    >>>
    >>> # In your main function
    >>> def main(cfg: MyConfig):
    ...     init_wandb_if_configured(cfg.wandb)
    ...     # Now kit.init() has been called with wandb backend
    ...     artifact.save(name="my-artifact")  # Will track in W&B
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    """Weights & Biases configuration for experiment tracking and artifact storage.

    When project is set, enables W&B artifact tracking. All fields are optional
    to support both tracked and untracked runs.

    Example CLI usage:
        nemotron nano3 data prep pretrain --wandb.project my-project --wandb.entity my-team
    """

    project: str | None = None
    """W&B project name (required to enable tracking)"""

    entity: str | None = None
    """W&B entity/team name"""

    run_name: str | None = None
    """W&B run name (auto-generated if not specified)"""

    tags: tuple[str, ...] = ()
    """Tags for filtering runs"""

    notes: str | None = None
    """Notes/description for the run"""

    @property
    def enabled(self) -> bool:
        """Returns True if wandb is configured (project is set)."""
        return self.project is not None


def init_wandb_if_configured(
    wandb_config: WandbConfig | None,
    job_type: str = "data-prep",
    tags: list[str] | None = None,
) -> None:
    """Initialize kit with wandb backend if WandbConfig is provided and enabled.

    This should be called at the start of command handlers to enable artifact tracking.
    If wandb_config is None or project is not set, this is a no-op.

    Args:
        wandb_config: WandbConfig instance or None
        job_type: W&B job type for categorizing runs (default: "data-prep")
        tags: Additional tags to add to the run (merged with config tags)

    Example:
        >>> def main(cfg: MyConfig):
        ...     init_wandb_if_configured(cfg.wandb, job_type="training", tags=["pretrain"])
        ...     # Artifacts will now be tracked in W&B
    """
    if wandb_config is None or not wandb_config.enabled:
        return

    import nemotron.kit as kit

    # Initialize kit with wandb backend (enables artifact tracking)
    kit.init(
        backend="wandb",
        wandb_project=wandb_config.project,
        wandb_entity=wandb_config.entity,
    )

    # Initialize wandb run if not already active
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb is required for W&B tracking. Install with: pip install wandb"
        )

    if wandb.run is None:
        # Merge config tags with additional tags
        all_tags: list[str] = []
        if wandb_config.tags:
            all_tags.extend(wandb_config.tags)
        if tags:
            all_tags.extend(tags)

        wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=wandb_config.run_name,
            tags=all_tags if all_tags else None,
            notes=wandb_config.notes,
            job_type=job_type,
        )


def add_wandb_tags(tags: list[str]) -> None:
    """Add tags to the active wandb run if one exists.

    This can be called after wandb is initialized to add stage-specific tags.
    Tags are merged with any existing tags on the run.

    Args:
        tags: List of tags to add to the run

    Example:
        >>> add_wandb_tags(["data-prep", "pretrain"])
    """
    try:
        import wandb

        if wandb.run is not None and tags:
            # Get existing tags and merge
            existing_tags = list(wandb.run.tags) if wandb.run.tags else []
            new_tags = list(set(existing_tags + tags))  # Deduplicate
            wandb.run.tags = new_tags
    except ImportError:
        pass
    except Exception:
        pass  # Don't fail if tags can't be added


def finish_wandb(exit_code: int = 0) -> None:
    """Finish the active wandb run if one exists.

    This should be called at the end of a successful run to properly close
    the wandb session. Without this, runs will appear as "crashed" in the
    W&B dashboard.

    Args:
        exit_code: Exit code to report. 0 for success, non-zero for failure.

    Example:
        >>> try:
        ...     # Do work
        ...     artifact.save()
        ...     finish_wandb(exit_code=0)
        ... except Exception:
        ...     finish_wandb(exit_code=1)
        ...     raise
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)
    except ImportError:
        pass


_HTTP_HANDLER_PATCHED = False
_WANDB_INIT_PATCHED = False
_LINEAGE_REGISTERED = False
_PENDING_ARTIFACT_QUALIFIED_NAMES: set[str] = set()
_PENDING_TAGS: set[str] = set()


def patch_wandb_http_handler_skip_digest_verification() -> None:
    """Best-effort patch to skip digest verification for HTTP reference artifacts.

    Some reference artifact backends (e.g. HuggingFace URLs) can return varying ETags
    over time, causing W&B to reject downloads due to digest mismatch.
    """
    global _HTTP_HANDLER_PATCHED
    if _HTTP_HANDLER_PATCHED:
        return

    try:
        from wandb.sdk.artifacts.storage_handlers import http_handler

        original_load_path = http_handler.HTTPHandler.load_path

        def patched_load_path(self, manifest_entry, local: bool = False):
            import os
            import tempfile

            import requests

            url = getattr(manifest_entry, "ref", None)
            if url is None:
                return original_load_path(self, manifest_entry, local=local)

            path = getattr(manifest_entry, "path", None)
            if local or path is None:
                fd, tmp_path = tempfile.mkstemp()
                os.close(fd)
                path = tmp_path

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return path

        http_handler.HTTPHandler.load_path = patched_load_path
        _HTTP_HANDLER_PATCHED = True
        logger.debug("Patched wandb HTTP handler to skip digest verification")
    except Exception as e:
        logger.warning(f"Failed to patch wandb HTTP handler: {e}")


def patch_wandb_init_for_lineage(
    *,
    artifact_qualified_names: list[str],
    tags: list[str] | None = None,
) -> None:
    """Patch `wandb.init()` so that, once a run is active, lineage is registered.

    Intended for setups where another library owns `wandb.init()` (e.g. Megatron-Bridge)
    but this project resolves artifacts before that init happens.
    """
    global _WANDB_INIT_PATCHED

    if artifact_qualified_names:
        _PENDING_ARTIFACT_QUALIFIED_NAMES.update(map(str, artifact_qualified_names))
    if tags:
        _PENDING_TAGS.update(map(str, tags))

    if _WANDB_INIT_PATCHED:
        return

    import wandb

    original_init = wandb.init

    def patched_init(*args, **kwargs):
        result = original_init(*args, **kwargs)
        _register_lineage_if_possible()
        return result

    wandb.init = patched_init
    _WANDB_INIT_PATCHED = True
    logger.debug("Patched wandb.init for lineage registration")


def _register_lineage_if_possible() -> None:
    global _LINEAGE_REGISTERED
    if _LINEAGE_REGISTERED:
        return

    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    if _PENDING_TAGS:
        add_wandb_tags(sorted(_PENDING_TAGS))

    for qname in sorted(_PENDING_ARTIFACT_QUALIFIED_NAMES):
        try:
            wandb.run.use_artifact(qname)
        except Exception as e:
            logger.warning(f"Failed to register artifact lineage for {qname}: {e}")

    _LINEAGE_REGISTERED = True
