"""Stub for Nemotron Next 3B v2 kwargs TypedDict.

Provides a lightweight TypedDict for local CLI development
when megatron-bridge isn't installed (requires CUDA to build).

This allows the CLI to show all kwargs options without needing
a full GPU environment.

Usage:
    The train.py will automatically use this stub when
    megatron-bridge import fails.
"""

from typing import Literal, Optional, TypedDict, Union


# Stub types for megatron-bridge types
class MixedPrecisionConfig:
    """Stub for megatron.bridge.training.config.MixedPrecisionConfig."""

    pass


class CommOverlapConfig:
    """Stub for megatron.bridge.training.config.CommOverlapConfig."""

    pass


# Provider literal type
NemotronNanoNext3Bv2Provider = Literal["local", "nemo", "te"]


class NemotronNext3Bv2CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Nemotron Next 3B v2 recipe helper functions.

    This TypedDict defines all the keyword arguments that can be passed to
    the nemotron_next_3b_v2_pretrain_config() recipe function.
    """

    # Core identifiers
    model_provider: NemotronNanoNext3Bv2Provider
    dir: Optional[str]
    name: str

    # Dataset configuration
    data_paths: Optional[list[str]]
    data_args_path: Optional[str]
    train_data_path: Optional[list[str]]
    valid_data_path: Optional[list[str]]
    test_data_path: Optional[list[str]]
    per_split_data_args_path: Optional[str]
    path_to_cache: Optional[str]
    mock: bool

    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_parallelism_dtype: Optional[str]  # torch.dtype as string for stub
    virtual_pipeline_parallelism: Optional[int]
    context_parallelism: int
    sequence_parallelism: bool
    expert_tensor_parallelism: int
    expert_model_parallelism: int

    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]

    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]

    # MoE
    enable_deepep: bool
