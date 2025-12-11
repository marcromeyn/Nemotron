"""Data preparation for Nano3 pretraining stage.

Tokenizes raw text data into Megatron bin/idx format.

Outputs blend.json with {"train": [...], "valid": [...], "test": [...]} format
compatible with Megatron-Bridge's per_split_data_args_path parameter.

Usage:
    python -m nemotron.recipes.nano3.stage0_pretrain.data_prep [options]
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from nemotron.data_prep import DataPrepConfig, PerSplitConfig, run_data_prep
from nemotron.kit import DataBlendsArtifact, cli, print_step_complete
from nemotron.kit.wandb import add_wandb_tags

STAGE_PATH = Path(__file__).parent

# Use NEMO_RUN_DIR for output when running via nemo-run (avoids writing to code dir)
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


@dataclass
class PreTrainDataPrepConfig:
    """Pretrain data preparation config.

    Tokenizes text into Megatron bin/idx format for pretraining.
    Outputs {"train": [...], "valid": [...], "test": [...]} JSON format.
    """

    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "data_blend_raw.json")
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "output/nano3/stage0_pretrain")
    """Output directory for tokenized data"""

    num_shards: int = 128
    """Number of output shards for parallel loading"""

    valid_shards: int = 1
    """Number of shards for validation split"""

    test_shards: int = 1
    """Number of shards for test split"""

    tokenizer_model: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    """HuggingFace tokenizer model name"""

    add_bos: bool = False
    """Prepend BOS token to documents"""

    add_eos: bool = True
    """Append EOS token to documents"""

    text_field: str = "text"
    """Default text field name in datasets"""

    min_doc_chars: int | None = None
    """Skip documents shorter than this"""

    max_doc_tokens: int | None = None
    """Truncate documents longer than this"""

    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    num_actors: int | None = None
    """Ray actors for parallel processing (None = auto)"""

    force: bool = False
    """Force new run, ignoring cache"""

    def __post_init__(self) -> None:
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"


def main(cfg: PreTrainDataPrepConfig) -> DataBlendsArtifact:
    """Run pretrain data preparation."""
    # Add stage-specific tags to wandb run
    add_wandb_tags(["data-prep", "pretrain"])

    # Build artifact name (e.g., "nano3/pretrain/data" or "nano3/pretrain/data?sample=100")
    artifact_name = f"nano3/pretrain/data{'?sample=' + str(cfg.sample) if cfg.sample else ''}"

    data_prep_config = DataPrepConfig(
        blend_path=cfg.blend_path,
        output_dir=cfg.output_dir,
        num_shards=cfg.num_shards,
        per_split=PerSplitConfig(
            enabled=True,
            valid_shards=cfg.valid_shards,
            test_shards=cfg.test_shards,
        ),
        tokenizer_model=cfg.tokenizer_model,
        add_bos=cfg.add_bos,
        add_eos=cfg.add_eos,
        text_field=cfg.text_field,
        min_doc_chars=cfg.min_doc_chars,
        max_doc_tokens=cfg.max_doc_tokens,
        sample=cfg.sample,
        num_actors=cfg.num_actors,
        force=cfg.force,
        artifact_name=artifact_name,
    )
    artifact = run_data_prep(data_prep_config)
    print_step_complete(data_prep=artifact)
    return artifact


if __name__ == "__main__":
    cli(main, ray=True)
