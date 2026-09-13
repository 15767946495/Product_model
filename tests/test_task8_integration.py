import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.task8_integration as task8
from tools.task8_integration import (
    atomic_exchange_directories,
    audit_staging,
    install_runtime_bundle,
    install_runtime_bundle_with_entrypoints,
    sha256,
)


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


def test_install_failure_keeps_old_bundle_unchanged(tmp_path, monkeypatch):
    active = tmp_path / "task8-active"
    bundle = tmp_path / "bundle"
    active.mkdir()
    bundle.mkdir()
    (active / "dataset.jsonl").write_text("old", encoding="utf-8")
    (bundle / "dataset.jsonl").write_text("new", encoding="utf-8")
    old_active = active / "dataset.jsonl"

    def fail_exchange(staging, target):
        raise OSError("injected exchange failure")

    monkeypatch.setattr(task8, "atomic_exchange_directories", fail_exchange)
    with pytest.raises(OSError, match="injected exchange failure"):
        install_runtime_bundle(bundle, active)

    assert old_active.read_text(encoding="utf-8") == "old"
    assert (bundle / "dataset.jsonl").read_text(encoding="utf-8") == "new"
    assert not (tmp_path / "task8-backup").exists()


def test_runtime_entrypoints_resolve_to_same_active_bundle(tmp_path):
    active = tmp_path / "task8-active"
    (active / "train_dataset").mkdir(parents=True)
    (active / "manifests").mkdir()
    task8.ensure_runtime_entrypoints(tmp_path, active)

    assert (tmp_path / "train_dataset").resolve().parent == active
    assert (tmp_path / "DataSrc/mmst_vit/manifests").resolve().parent == active
    assert (tmp_path / "train_dataset").resolve().parent == (tmp_path / "DataSrc/mmst_vit/manifests").resolve().parent


def test_install_failure_keeps_both_default_entrypoints_on_old_bundle(tmp_path, monkeypatch):
    active = tmp_path / "task8-active"
    bundle = tmp_path / "bundle"
    (active / "train_dataset").mkdir(parents=True)
    (active / "manifests").mkdir()
    bundle.mkdir()
    (active / "train_dataset" / "dataset.jsonl").write_text("old-data", encoding="utf-8")
    (active / "manifests" / "valid-no-ia.jsonl").write_text("old-manifest", encoding="utf-8")
    (bundle / "train_dataset").mkdir()
    (bundle / "manifests").mkdir()
    (bundle / "train_dataset" / "dataset.jsonl").write_text("new-data", encoding="utf-8")
    (bundle / "manifests" / "valid-no-ia.jsonl").write_text("new-manifest", encoding="utf-8")
    task8.ensure_runtime_entrypoints(tmp_path, active)

    monkeypatch.setattr(task8, "atomic_exchange_directories", lambda *_: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError, match="injected"):
        install_runtime_bundle_with_entrypoints(bundle, tmp_path)

    assert (tmp_path / "train_dataset/dataset.jsonl").read_text(encoding="utf-8") == "old-data"
    assert (tmp_path / "DataSrc/mmst_vit/manifests/valid-no-ia.jsonl").read_text(encoding="utf-8") == "old-manifest"
    assert (tmp_path / "train_dataset").resolve().parent == (tmp_path / "DataSrc/mmst_vit/manifests").resolve().parent


def test_root_valid_audit_rejects_duplicate_and_count_mismatch(tmp_path):
    staging = tmp_path / "staging"
    manifests = staging / "manifests"
    manifests.mkdir(parents=True)
    row = {"FIPS": "17001", "Year": 2020, "State": "illinois", "County": "a",
           "month": [4] * 28 + [5] * 28 + [6] * 28 + [7] * 28 + [8] * 28 + [9] * 28,
           "day": list(range(1, 29)) * 6, "l_enc": 168}
    (staging / "dataset.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (manifests / "valid-no-ia.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )
    for split in ("train", "val", "test"):
        (manifests / f"valid-no-ia.{split}.jsonl").write_text("\n", encoding="utf-8")
        (manifests / f"{split}.official.no-ia.json").write_text("[]", encoding="utf-8")
    import torch
    entry = {"feats": torch.zeros(1, 168, 12), "coords": torch.zeros(1, 2),
             "month": torch.tensor(row["month"]), "day": torch.tensor(row["day"]), "l_enc": 168}
    torch.save({"version": 4, "coord_type": "grid_center", "time_window": "04-01--09-28",
                "days_per_month": 28, "max_steps": 168, "entries": [entry]}, staging / "grid_cache.pt")
    (staging / "grid_cache_meta.json").write_text(
        json.dumps({"version": 4, "time_window": "04-01--09-28", "days_per_month": 28, "max_steps": 168}),
        encoding="utf-8"
    )
    result = audit_staging(staging, tmp_path, manifests)
    assert "root valid-no-ia contains duplicate identity keys" in result["violations"]
    assert "root valid-no-ia row count differs from shared JSONL" in result["violations"]


def test_staging_audit_checks_official_structure_state_and_paths(tmp_path):
    staging = tmp_path / "staging"
    manifests = staging / "manifests"
    manifests.mkdir(parents=True)
    row = {"FIPS": "17001", "Year": 2020, "State": "illinois", "County": "a",
           "month": [4] * 28 + [5] * 28 + [6] * 28 + [7] * 28 + [8] * 28 + [9] * 28,
           "day": list(range(1, 29)) * 6, "l_enc": 168}
    (staging / "dataset.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    for filename in ("valid-no-ia.jsonl", "valid-no-ia.train.jsonl"):
        (manifests / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")
    for split in ("val", "test"):
        (manifests / f"valid-no-ia.{split}.jsonl").write_text("", encoding="utf-8")
    malformed = {"FIPS": "17001", "year": 2020, "state": "iowa", "data": {}}
    for split in ("train", "val", "test"):
        (manifests / f"{split}.official.no-ia.json").write_text(json.dumps([malformed]), encoding="utf-8")
    import torch
    entry = {"feats": torch.zeros(1, 168, 12), "coords": torch.zeros(1, 2),
             "month": torch.tensor(row["month"]), "day": torch.tensor(row["day"]), "l_enc": 168}
    torch.save({"version": 4, "coord_type": "grid_center", "time_window": "04-01--09-28",
                "days_per_month": 28, "max_steps": 168, "entries": [entry]}, staging / "grid_cache.pt")
    (staging / "grid_cache_meta.json").write_text(
        json.dumps({"version": 4, "time_window": "04-01--09-28", "days_per_month": 28, "max_steps": 168}),
        encoding="utf-8"
    )
    result = audit_staging(staging, tmp_path, manifests)
    assert any("official record 0: state outside allowlist" in item for item in result["violations"])
    assert any("malformed structure" in item for item in result["violations"])


def test_audit_only_reads_active_bundle_without_installing(tmp_path, capsys):
    active = tmp_path / "task8-active"
    dataset = active / "train_dataset"
    manifests = active / "manifests"
    manifests.mkdir(parents=True)
    dataset.mkdir()
    (dataset / "dataset.jsonl").write_text("", encoding="utf-8")
    import torch
    torch.save({"version": 4, "coord_type": "grid_center", "time_window": "04-01--09-28",
                "days_per_month": 28, "max_steps": 168, "entries": []}, dataset / "grid_cache.pt")
    (dataset / "grid_cache_meta.json").write_text(
        json.dumps({"version": 4, "time_window": "04-01--09-28", "days_per_month": 28, "max_steps": 168}),
        encoding="utf-8"
    )
    (manifests / "valid-no-ia.jsonl").write_text("", encoding="utf-8")
    for split in ("train", "val", "test"):
        (manifests / f"valid-no-ia.{split}.jsonl").write_text("", encoding="utf-8")
        (manifests / f"{split}.official.no-ia.json").write_text("[]", encoding="utf-8")

    task8.main(["--audit-only", "--runtime-root", str(tmp_path), "--data-root", str(tmp_path)])

    assert json.loads(capsys.readouterr().out)["rows"] == 0
    assert not (tmp_path / "task8-report.json").exists()


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


def test_backup_move_failure_rolls_back_active_and_keeps_existing_backup(tmp_path, monkeypatch):
    active = tmp_path / "task8-active"
    bundle = tmp_path / "bundle"
    backup = tmp_path / "task8-backup"
    for root, value in ((active, "old"), (bundle, "new"), (backup, "older")):
        (root / "train_dataset").mkdir(parents=True)
        (root / "manifests").mkdir()
        (root / "train_dataset/dataset.jsonl").write_text(value, encoding="utf-8")
        (root / "manifests/valid-no-ia.jsonl").write_text(value, encoding="utf-8")
    task8.ensure_runtime_entrypoints(tmp_path, active)
    real_replace = task8.os.replace

    def fail_backup_move(source, destination):
        if Path(source) == bundle and Path(destination).name.startswith("task8-backup."):
            raise OSError("injected backup move failure")
        return real_replace(source, destination)

    monkeypatch.setattr(task8.os, "replace", fail_backup_move)
    with pytest.raises(OSError, match="backup move failure"):
        install_runtime_bundle_with_entrypoints(bundle, tmp_path)

    assert (tmp_path / "train_dataset/dataset.jsonl").read_text(encoding="utf-8") == "old"
    assert (tmp_path / "DataSrc/mmst_vit/manifests/valid-no-ia.jsonl").read_text(encoding="utf-8") == "old"
    assert (backup / "train_dataset/dataset.jsonl").read_text(encoding="utf-8") == "older"
    assert (bundle / "train_dataset/dataset.jsonl").read_text(encoding="utf-8") == "new"


def test_entrypoint_creation_failure_rolls_back_new_active_and_created_link(tmp_path, monkeypatch):
    active = tmp_path / "task8-active"
    bundle = tmp_path / "bundle"
    for root, value in ((active, "old"), (bundle, "new")):
        (root / "train_dataset").mkdir(parents=True)
        (root / "manifests").mkdir()
        (root / "train_dataset/dataset.jsonl").write_text(value, encoding="utf-8")
        (root / "manifests/valid-no-ia.jsonl").write_text(value, encoding="utf-8")
    real_symlink = task8.os.symlink
    calls = 0

    def fail_second_link(source, destination, target_is_directory=False):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected entrypoint failure")
        return real_symlink(source, destination, target_is_directory=target_is_directory)

    monkeypatch.setattr(task8.os, "symlink", fail_second_link)
    with pytest.raises(OSError, match="entrypoint failure"):
        install_runtime_bundle_with_entrypoints(bundle, tmp_path)

    assert (active / "train_dataset/dataset.jsonl").read_text(encoding="utf-8") == "old"
    assert (bundle / "train_dataset/dataset.jsonl").read_text(encoding="utf-8") == "new"
    assert not (tmp_path / "train_dataset").exists()
    assert not (tmp_path / "DataSrc/mmst_vit/manifests").exists()


def test_missing_active_install_failure_restores_original_missing_state(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "train_dataset").mkdir(parents=True)
    (bundle / "manifests").mkdir()
    (bundle / "train_dataset/dataset.jsonl").write_text("new", encoding="utf-8")
    real_symlink = task8.os.symlink

    def fail_first_link(*args, **kwargs):
        raise OSError("injected first install failure")

    monkeypatch.setattr(task8.os, "symlink", fail_first_link)
    with pytest.raises(OSError, match="first install failure"):
        install_runtime_bundle_with_entrypoints(bundle, tmp_path)

    assert not (tmp_path / "task8-active").exists()
    assert not (tmp_path / "train_dataset").exists()
    assert not (tmp_path / "DataSrc/mmst_vit/manifests").exists()
    assert (bundle / "train_dataset/dataset.jsonl").read_text(encoding="utf-8") == "new"
    monkeypatch.setattr(task8.os, "symlink", real_symlink)


@pytest.mark.parametrize("relative", ["/tmp/outside", "../outside"])
def test_official_audit_rejects_paths_outside_data_root(tmp_path, relative):
    record = {
        "state": "illinois",
        "data": {
            "USDA": relative,
            "HRRR": {"short_term": [relative] * 6, "long_term": [[relative] * 60]},
            "sentinel": [relative] * 4,
        },
    }
    violations = task8._validate_official_record(record, tmp_path / "data", 0)
    assert any("path escapes data_root" in item for item in violations)


def test_staging_audit_rejects_fractional_l_enc(tmp_path):
    staging = tmp_path / "staging"
    manifests = staging / "manifests"
    manifests.mkdir(parents=True)
    month = [4] * 28 + [5] * 28 + [6] * 28 + [7] * 28 + [8] * 28 + [9] * 28
    row = {"FIPS": "17001", "Year": 2020, "State": "illinois", "County": "a",
           "month": month, "day": list(range(1, 29)) * 6, "l_enc": 168.9}
    (staging / "dataset.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (manifests / "valid-no-ia.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    for split in ("train", "val", "test"):
        (manifests / f"valid-no-ia.{split}.jsonl").write_text("", encoding="utf-8")
        (manifests / f"{split}.official.no-ia.json").write_text("[]", encoding="utf-8")
    import torch
    entry = {"feats": torch.zeros(1, 168, 12), "coords": torch.zeros(1, 2),
             "month": torch.tensor(month), "day": torch.tensor(row["day"]), "l_enc": 168}
    torch.save({"version": 4, "coord_type": "grid_center", "time_window": "04-01--09-28",
                "days_per_month": 28, "max_steps": 168, "entries": [entry]}, staging / "grid_cache.pt")
    (staging / "grid_cache_meta.json").write_text(
        json.dumps({"version": 4, "time_window": "04-01--09-28", "days_per_month": 28, "max_steps": 168}),
        encoding="utf-8"
    )
    result = audit_staging(staging, tmp_path, manifests)
    assert "dataset row 0: l_enc is not 168" in result["violations"]
