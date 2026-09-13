"""Pinned official MMST-ViT source management and command construction."""

from __future__ import annotations

import subprocess
import json
from pathlib import Path
from collections.abc import Iterable, Mapping

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


def _record_paths(record: Mapping) -> list[str]:
    weather = record["data"]["HRRR"]
    return [
        record["data"]["USDA"],
        *weather["short_term"],
        *(path for context in weather["long_term"] for path in context),
        *record["data"]["sentinel"],
    ]


def validate_official_records(records: Iterable[Mapping], data_root: Path) -> None:
    missing = sorted(
        str(data_root / relative)
        for record in records
        for relative in _record_paths(record)
        if not (data_root / relative).is_file()
    )
    if missing:
        raise FileNotFoundError("official manifest paths missing:\n" + "\n".join(missing))


def write_official_split(path: Path, records: list[Mapping], data_root: Path) -> None:
    validate_official_records(records, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_official_manifests(samples: Iterable[Mapping], output_dir: Path, data_root: Path) -> dict[str, int]:
    from mmst_vit.config import official_sample_record
    from mmst_vit.manifest import split_samples

    sample_list = [dict(sample) for sample in samples]
    records = {
        (sample["FIPS"], int(sample["Year"])): official_sample_record(
            sample["FIPS"], sample["Year"], sample["State"], sample["County"], data_root
        )
        for sample in sample_list
    }
    counts = {}
    for name, split in split_samples(sample_list).items():
        split_records = [records[(row["FIPS"], int(row["Year"]))] for row in split]
        write_official_split(output_dir / f"{name}.official.no-ia.json", split_records, data_root)
        counts[name] = len(split_records)
    return counts
