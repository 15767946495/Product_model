import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.task8_integration import atomic_exchange_directories, sha256


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
