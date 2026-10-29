"""统一运行输出（规范 11.4）：output_dir/<model>/<run_id>/ 下的固定文件与 run_metadata 字段。

所有基线共用；只负责落盘与元数据，不涉及训练逻辑。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_CHECKPOINT = "checkpoint-final.pth"
LATEST_CHECKPOINT = "checkpoint-latest.pth"
RUN_FILES = ("config.json", LATEST_CHECKPOINT, FINAL_CHECKPOINT, "train_metrics.jsonl",
             "test_metrics.json", "county_predictions.jsonl", "run_metadata.json")
PREDICTION_FIELDS = ("sample_id", "fips", "year", "y_true", "y_pred")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def write_json(path, payload) -> None:
    """原子写入，中断不会留下半个文件。"""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_json(path, **fields) -> None:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(fields)
    write_json(path, payload)


def append_jsonl(path, row) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=_jsonable) + "\n")


def sha256_file(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(*, model_name, adaptation, dataset_jsonl, train_years, test_years,
                        model_config) -> dict:
    return {
        "model_name": str(model_name),
        "adaptation": str(adaptation),
        "dataset_sha256": sha256_file(Path(dataset_jsonl)),
        "train_years": list(train_years),
        "test_years": list(test_years),
        "model_config": dict(model_config),
    }


def git_commit(root: Path = PROJECT_ROOT) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def make_run_dir(output_root, model_name: str, run_id: str | None) -> Path:
    run_id = run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def count_parameters(model) -> dict[str, int]:
    params = list(model.parameters())
    return {"total": sum(p.numel() for p in params),
            "trainable": sum(p.numel() for p in params if p.requires_grad)}


def base_metadata(*, model: str, adaptation: str, run_id: str, dataset_jsonl, seed: int,
                  train_years, test_years, counties: int, inputs, **extra) -> dict:
    """规范 11.4 要求的 run_metadata 字段；训练结束后再补 epoch、参数量和 checkpoint SHA-256。"""
    dataset_jsonl = Path(dataset_jsonl)
    return {
        "model": model, "adaptation": adaptation, "run_id": run_id,
        "git_commit": git_commit(), "dataset_jsonl": str(dataset_jsonl),
        "dataset_sha256": sha256_file(dataset_jsonl), "seed": seed,
        "train_years": list(train_years), "test_years": list(test_years),
        "counties": counties, "inputs": list(inputs),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started": now(), "status": "running", **extra,
    }


def write_predictions(run_dir, rows, expected_sample_ids) -> None:
    """逐县预测按统一 Dataset 的 2022 sample_id 顺序写出；顺序或集合不符直接失败。"""
    ids = [row["sample_id"] for row in rows]
    if ids != list(expected_sample_ids):
        raise RuntimeError("county predictions do not follow unified Dataset sample order")
    path = Path(run_dir) / "county_predictions.jsonl"
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    for row in rows:
        append_jsonl(tmp, {k: row[k] for k in PREDICTION_FIELDS})
    os.replace(tmp, path)


def save_checkpoint(path, payload) -> Path:
    """原子写 checkpoint，中断不会留下损坏文件。"""
    import torch

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def require_final_checkpoint(checkpoint_path) -> None:
    """最终测试只接受 checkpoint-final.pth（不按任何指标挑选 checkpoint）。"""
    if Path(checkpoint_path).name != FINAL_CHECKPOINT:
        raise ValueError(f"final test only evaluates {FINAL_CHECKPOINT}, got {checkpoint_path}")


def _checkpoint_field(checkpoint: dict, field: str):
    value = checkpoint
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"checkpoint.{field}: missing field")
        value = value[part]
    return value


def _validate_final_checkpoint(checkpoint_path, *, model_name, adaptation, dataset_jsonl,
                               train_years, test_years, required_fields) -> dict:
    import torch

    require_final_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint: expected dict payload")
    if not isinstance(checkpoint.get("model_config"), dict):
        raise ValueError("checkpoint.model_config: expected dict")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("checkpoint.model: expected Mapping")
    expected = checkpoint_metadata(
        model_name=model_name,
        adaptation=adaptation,
        dataset_jsonl=dataset_jsonl,
        train_years=train_years,
        test_years=test_years,
        model_config=checkpoint["model_config"],
    )
    for field in ("model_name", "adaptation", "dataset_sha256", "train_years", "test_years"):
        if checkpoint.get(field) != expected[field]:
            raise ValueError(
                f"checkpoint.{field}: expected {expected[field]!r}, got {checkpoint.get(field)!r}"
            )
    for field in required_fields:
        _checkpoint_field(checkpoint, field)
    return checkpoint


def validate_final_checkpoint(checkpoint_path, *, model_name, adaptation, dataset_jsonl,
                              train_years, test_years, required_fields: tuple[str, ...] = ()) -> dict:
    checkpoint = _validate_final_checkpoint(
        checkpoint_path,
        model_name=model_name,
        adaptation=adaptation,
        dataset_jsonl=dataset_jsonl,
        train_years=train_years,
        test_years=test_years,
        required_fields=required_fields,
    )
    print(f"Loaded final checkpoint: {Path(checkpoint_path).resolve()}")
    return checkpoint


def validate_unified_run(checkpoint_path, output_dir, *, model_name, adaptation, dataset_jsonl,
                         train_years, test_years, required_fields: tuple[str, ...] = (),
                         log_loaded: bool = True) -> tuple[dict, dict]:
    """校验监督模型 final checkpoint 所属的统一 run 目录及 metadata。"""
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent
    checkpoint = _validate_final_checkpoint(
        checkpoint_path,
        model_name=model_name,
        adaptation=adaptation,
        dataset_jsonl=dataset_jsonl,
        train_years=train_years,
        test_years=test_years,
        required_fields=required_fields,
    )
    metadata_path = run_dir / "run_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "unified run directory: run_metadata must be a readable JSON object"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata: expected JSON object")
    expected = {
        "model": checkpoint["model_name"],
        "adaptation": checkpoint["adaptation"],
        "dataset_sha256": checkpoint["dataset_sha256"],
        "train_years": checkpoint["train_years"],
        "test_years": checkpoint["test_years"],
        "run_id": run_dir.name,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"run_metadata.{field}: expected {value!r}, got {metadata.get(field)!r}")
    if "status" not in metadata:
        raise ValueError("run_metadata.status: missing field")
    if log_loaded:
        print(f"Loaded final checkpoint: {checkpoint_path}")
    return checkpoint, metadata


def finish_test(run_dir, rows, expected_sample_ids, *, checkpoint_path, years) -> dict:
    """写 test_metrics.json、county_predictions.jsonl，并在 run_metadata 记录测试与 checkpoint SHA-256。"""
    from train_dataset.metrics import regression_metrics

    run_dir = Path(run_dir)
    write_predictions(run_dir, rows, expected_sample_ids)
    metrics = regression_metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
    write_json(run_dir / "test_metrics.json",
               {**metrics, "years": list(years), "checkpoint": str(checkpoint_path)})
    update_json(run_dir / "run_metadata.json", status="finished", evaluated_years=list(years),
                test_samples=len(rows), test_checkpoint=str(checkpoint_path),
                checkpoint_sha256=sha256_file(checkpoint_path), finished=now())
    print("Final test {}: RMSE {:.2f}  MAE {:.2f}  R2 {:.3f}  Corr {:.3f}".format(
        list(years), metrics["rmse"], metrics["mae"], metrics["r2"], metrics["corr"]))
    return metrics


def add_common_args(
    parser,
    *,
    epochs: int,
    lr: float | None,
    seed: int = 0,
    output_root: Path | str | None = None,
):
    """统一数据 / 输出 / 训练基础参数；不接受 root_dir、manifest 或验证集参数。"""
    from train_dataset.dataset import DATA_ROOT

    parser.add_argument("--dataset-jsonl", dest="dataset_jsonl",
                        default=str(PROJECT_ROOT / "train_dataset" / "dataset.jsonl"),
                        help="unified CropNet JSONL (train_dataset/dataset.jsonl)")
    parser.add_argument("--data-root", dest="data_root", default=str(DATA_ROOT),
                        help="root that HDF5 paths in the JSONL are relative to")
    output_root = Path(PROJECT_ROOT / "output_dir" if output_root is None else output_root).resolve()
    parser.add_argument("--output_root", default=str(output_root),
                        help="runs go to output_root/<model>/<run_id>/")
    parser.add_argument("--run_id", default=None, help="defaults to a timestamp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--epochs", type=int, default=epochs, help="fixed epoch budget; no early stopping")
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--save_freq", type=int, default=5, help="epochs between checkpoint-latest.pth writes")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
