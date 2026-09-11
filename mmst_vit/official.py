"""Pinned official MMST-ViT source management and command construction."""

from __future__ import annotations

import subprocess
from pathlib import Path

OFFICIAL_REPOSITORY = "https://github.com/fudong03/MMST-ViT.git"
OFFICIAL_REVISION = "615666c8d9fcd704acb662c13065703cbf2eab70"


def ensure_official_source(source_dir: Path, repository: str = OFFICIAL_REPOSITORY, revision: str = OFFICIAL_REVISION) -> str:
    if not (source_dir / ".git").exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repository, str(source_dir)], check=True)
    subprocess.run(["git", "fetch", "--all", "--tags"], cwd=source_dir, check=True)
    subprocess.run(["git", "checkout", "--force", revision], cwd=source_dir, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_dir, text=True).strip()


def build_official_command(config_path: Path, mode: str) -> list[str]:
    if mode not in {"train", "evaluate"}:
        raise ValueError(f"unsupported mode: {mode}")
    import json

    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = [
        "python", "main_finetune_mmst_vit.py",
        "--root_dir", config["data_root"],
        "--data_file_train", config["train_file"],
        "--data_file_val", config["val_file"],
        "--output_dir", config["output_dir"],
        "--log_dir", config["output_dir"],
    ]
    if mode == "evaluate":
        args.append("--eval")
    return args
