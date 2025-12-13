# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from nemo_run.core.packaging import Packager, PatternPackager

from nemotron.kit.packaging.ast_inliner import inline_imports


@dataclass(kw_only=True)
class SelfContainedPackager(Packager):
    """Packager that produces a self-contained `main.py` by inlining `nemotron.*` imports."""

    script_path: str
    train_path: Path
    inline_package: str = "nemotron"

    def package(self, path: Path, job_dir: str, name: str) -> str:
        repo_root = Path(path)
        output_file = os.path.join(job_dir, f"{name}.tar.gz")
        if os.path.exists(output_file):
            return output_file

        staging_dir = Path(job_dir) / "code"
        staging_dir.mkdir(parents=True, exist_ok=True)

        script_file = Path(self.script_path)
        if not script_file.is_absolute():
            script_file = repo_root / self.script_path

        inlined = inline_imports(
            script_file,
            repo_root=repo_root,
            package_prefix=self.inline_package,
        )
        (staging_dir / "main.py").write_text(inlined, encoding="utf-8")

        shutil.copy2(self.train_path, staging_dir / "config.yaml")

        packager = PatternPackager(
            include_pattern=[str(staging_dir / "main.py"), str(staging_dir / "config.yaml")],
            relative_path=[str(staging_dir), str(staging_dir)],
        )

        # On macOS, bsdtar can include AppleDouble `._*` entries unless disabled.
        prev = os.environ.get("COPYFILE_DISABLE")
        os.environ["COPYFILE_DISABLE"] = "1"
        try:
            return packager.package(repo_root, job_dir, name)
        finally:
            if prev is None:
                os.environ.pop("COPYFILE_DISABLE", None)
            else:
                os.environ["COPYFILE_DISABLE"] = prev


__all__ = ["SelfContainedPackager"]
