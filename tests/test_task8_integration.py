import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.task8_integration import atomic_exchange_directories, audit_staging, install_runtime_bundle, sha256


def test_atomic_exchange_replaces_complete_directory(tmp_path):
    active = tmp_path / "active"
    staging = tmp_path / "staging"
    active.mkdir()
    staging.mkdir()
    (active / "dataset.jsonl").write_text("old", encoding="utf-8")
    for name in ("dataset.jsonl", "grid_cache.pt", "grid_cache_meta.json"):
        (staging / name).write_text("new", encoding="utf-8")

    atomic_exchange_directories(staging, active)

    assert {path.name for path in active.iterdir()} == {
        "dataset.jsonl", "grid_cache.pt", "grid_cache_meta.json"
    }
    assert (active / "dataset.jsonl").read_text(encoding="utf-8") == "new"
    assert (staging / "dataset.jsonl").read_text(encoding="utf-8") == "old"


def test_sha256_reports_file_content(tmp_path):
    path = tmp_path / "value"
    path.write_text("abc", encoding="utf-8")
    assert sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_install_runtime_bundle_keeps_previous_bundle_as_rollback(tmp_path):
    active = tmp_path / "task8-active"
    bundle = tmp_path / "bundle"
    active.mkdir()
    bundle.mkdir()
    (active / "valid-no-ia.jsonl").write_text("old", encoding="utf-8")
    (bundle / "valid-no-ia.jsonl").write_text("new", encoding="utf-8")

    install_runtime_bundle(bundle, active)

    assert (active / "valid-no-ia.jsonl").read_text(encoding="utf-8") == "new"
    assert (tmp_path / "task8-backup" / "valid-no-ia.jsonl").read_text(encoding="utf-8") == "old"


def test_staging_audit_requires_root_valid_manifest_and_meta_days(tmp_path):
    staging = tmp_path / "staging"
    manifests = staging / "manifests"
    manifests.mkdir(parents=True)
    row = {"FIPS": "17001", "Year": 2020, "State": "illinois", "County": "a"}
    (staging / "dataset.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (manifests / "valid-no-ia.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    for split, records in (("train", [row]), ("val", []), ("test", [])):
        (manifests / f"valid-no-ia.{split}.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
        )
        (manifests / f"{split}.official.no-ia.json").write_text("[]", encoding="utf-8")
    import torch
    entry = {"feats": torch.zeros(1, 168, 12), "coords": torch.zeros(1, 2),
             "month": torch.tensor([4] * 28 + [5] * 28 + [6] * 28 + [7] * 28 + [8] * 28 + [9] * 28),
             "day": torch.tensor(list(range(1, 29)) * 6), "l_enc": 168}
    torch.save({"version": 4, "coord_type": "grid_center", "time_window": "04-01--09-28", "days_per_month": 28,
                "max_steps": 168, "entries": [entry]}, staging / "grid_cache.pt")
    (staging / "grid_cache_meta.json").write_text(
        json.dumps({"version": 4, "time_window": "04-01--09-28", "days_per_month": 27, "max_steps": 168}),
        encoding="utf-8",
    )
    result = audit_staging(staging, tmp_path, manifests)
    assert "meta days_per_month mismatch" in result["violations"]
