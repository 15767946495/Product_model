"""正式 MMST-TFT 推理入口。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MMST_ROOT = ROOT / "MMST-ViT"
DEFAULT_DATA_ROOT = Path("/data/raid0/hqx/Product_model_runtime/DataSrc")
DEFAULT_EVAL_CONFIG = MMST_ROOT / "data/corn_val.json"
DEFAULT_SOIL_FILE = DEFAULT_DATA_ROOT / "soil/county_soil.json"
DEFAULT_CHECKPOINT = Path("/data/raid0/hqx/TFT_train/mmst_frontend_2021/checkpoint-final.pth")

from TFT_model.mmst_data import MMSTCornDataset
from TFT_model.mmst_tft_model import MMSTTFTYieldModel
from TFT_model.train_mmst_frontend import (
    DEFAULT_CHECKPOINT as FORMAL_PRETRAINED_CHECKPOINT,
    DEFAULT_TRAIN_CONFIG as FORMAL_TRAIN_CONFIG,
    zscore_soil,
)


def _load_eval_data_wrapper():
    sentinel_name = "_tft_mmst_official_sentinel_for_infer"
    wrapper_name = "_tft_mmst_official_data_wrapper_for_infer"
    sentinel_path = MMST_ROOT / "dataset/sentinel_loader.py"
    wrapper_path = MMST_ROOT / "dataset/data_wrapper.py"
    sentinel_spec = importlib.util.spec_from_file_location(sentinel_name, sentinel_path)
    if sentinel_spec is None or sentinel_spec.loader is None:
        raise ImportError(f"无法加载官方 Sentinel 模块: {sentinel_path}")
    sentinel_module = importlib.util.module_from_spec(sentinel_spec)
    sentinel_spec.loader.exec_module(sentinel_module)

    old_dataset = sys.modules.get("dataset")
    old_sentinel = sys.modules.get("dataset.sentinel_loader")
    package = type(sys)("dataset")
    package.__path__ = [str(MMST_ROOT / "dataset")]
    sys.modules["dataset"] = package
    sys.modules["dataset.sentinel_loader"] = sentinel_module
    try:
        wrapper_spec = importlib.util.spec_from_file_location(wrapper_name, wrapper_path)
        if wrapper_spec is None or wrapper_spec.loader is None:
            raise ImportError(f"无法加载官方 DataWrapper 模块: {wrapper_path}")
        wrapper_module = importlib.util.module_from_spec(wrapper_spec)
        wrapper_spec.loader.exec_module(wrapper_module)
        return wrapper_module.DataWrapper
    finally:
        if old_dataset is None:
            sys.modules.pop("dataset", None)
        else:
            sys.modules["dataset"] = old_dataset
        if old_sentinel is None:
            sys.modules.pop("dataset.sentinel_loader", None)
        else:
            sys.modules["dataset.sentinel_loader"] = old_sentinel


DataWrapper = _load_eval_data_wrapper()


def transform_eval_images(images: torch.Tensor) -> torch.Tensor:
    """Apply the official validation transform to raw HWC county images."""
    if images.ndim != 6 or images.shape[1] != 6 or images.shape[-1] != 3:
        raise ValueError("raw images must have shape (B, 6, G, H, W, 3)")
    batch, months, grids, height, width, channels = images.shape
    flat = images.reshape(batch * months * grids, height, width, channels)
    flat = flat.permute(0, 3, 1, 2).to(torch.float32).div(255.0)
    transformed, _ = DataWrapper(train=False)(flat)
    return transformed.reshape(batch, months, grids, 3, 224, 224)


def _metric_corr(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.size < 2 or np.std(actual) == 0.0 or np.std(predicted) == 0.0:
        return 0.0
    return float(np.corrcoef(actual, predicted)[0, 1])


def metrics_from_log_pairs(pairs):
    """Calculate county metrics after exponentiating log-yield values."""
    actual = np.asarray([math.exp(float(item["yield_log"])) for item in pairs], dtype=float)
    predicted = np.asarray([math.exp(float(item["log_mean"])) for item in pairs], dtype=float)
    if actual.size == 0:
        raise ValueError("cannot calculate metrics for an empty evaluation set")
    residual = predicted - actual
    total = ((actual - actual.mean()) ** 2).sum()
    return {
        "count": int(actual.size),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual ** 2) / total) if total else 0.0,
        "corr": _metric_corr(actual, predicted),
    }


def load_formal_checkpoint(path, model_factory=MMSTTFTYieldModel):
    path = Path(path)
    if path.name != "checkpoint-final.pth":
        raise ValueError("formal inference requires checkpoint-final.pth")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required_fields = {
        "model", "optimizer", "epoch", "config", "schedule",
        "soil_stats", "source_checkpoint",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError("checkpoint-final.pth has invalid formal fields")
    if not isinstance(payload["model"], dict) or not isinstance(payload["optimizer"], dict):
        raise ValueError("checkpoint model and optimizer must be dictionaries")
    if payload["epoch"] != 100:
        raise ValueError("checkpoint epoch must be 100")

    config = payload["config"]
    schedule = payload["schedule"]
    soil_stats = payload["soil_stats"]
    if not all(isinstance(value, dict) for value in (config, schedule, soil_stats)):
        raise ValueError("checkpoint config, schedule, and soil_stats must be dictionaries")
    expected_config = {
        "train_year": 2021,
        "epochs": 100,
        "outer_batch_size": 1,
        "hidden_size": 512,
        "heads": 4,
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
    }
    for name, expected in expected_config.items():
        if name not in config or config[name] != expected:
            raise ValueError(f"checkpoint config requires {name}={expected}")
    try:
        train_config = Path(config["train_config"]).resolve()
        pretrained = Path(config["pretrained"]).resolve()
        source_checkpoint = Path(payload["source_checkpoint"]).resolve()
    except (KeyError, TypeError) as error:
        raise ValueError("checkpoint paths are missing or invalid") from error
    if train_config != FORMAL_TRAIN_CONFIG.resolve():
        raise ValueError("checkpoint config requires canonical corn_train.json")
    if pretrained != FORMAL_PRETRAINED_CHECKPOINT.resolve():
        raise ValueError("checkpoint config requires canonical checkpoint-latest.pth")
    if source_checkpoint != pretrained:
        raise ValueError("source_checkpoint must resolve to config.pretrained")

    expected_schedule = {
        "name": "linear-warmup-cosine",
        "base_lr": 1e-3,
        "warmup_epochs": 5,
        "total_epochs": 100,
        "min_lr": 0.0,
    }
    if schedule != expected_schedule:
        raise ValueError("checkpoint schedule does not match the formal protocol")
    if not isinstance(soil_stats.get("mean"), list) or not isinstance(soil_stats.get("std"), list):
        raise ValueError("checkpoint soil_stats must contain mean and std lists")
    if len(soil_stats["mean"]) != 7 or len(soil_stats["std"]) != 7:
        raise ValueError("checkpoint soil_stats mean and std must each contain 7 values")
    if soil_stats.get("source") != {"year": 2021, "county_count": 195}:
        raise ValueError("checkpoint soil_stats source must be 195 counties from 2021")
    model = model_factory(
        config["pretrained"],
        hidden_size=int(config["hidden_size"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model, config, soil_stats


def build_parser():
    parser = argparse.ArgumentParser(description="正式 MMST-TFT Corn 239 县推理")
    parser.add_argument("--root-dir", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--eval-config", default=str(DEFAULT_EVAL_CONFIG))
    parser.add_argument("--soil-file", default=str(DEFAULT_SOIL_FILE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--predictions-json", default=None)
    parser.add_argument("--metrics-json", default=None)
    return parser


def evaluate(args):
    checkpoint = Path(args.checkpoint)
    model, config, soil_stats = load_formal_checkpoint(checkpoint)
    if len(json.loads(Path(args.eval_config).read_text(encoding="utf-8"))) != 239:
        raise ValueError("formal evaluation requires corn_val.json with 239 counties")
    mean = torch.tensor(soil_stats["mean"], dtype=torch.float32)
    std = torch.tensor(soil_stats["std"], dtype=torch.float32)
    device = torch.device(args.device)
    model = model.to(device).eval()
    dataset = MMSTCornDataset(args.root_dir, args.eval_config, args.soil_file)
    try:
        if len(dataset) != 239 or len(set(dataset.fips_codes)) != 239:
            raise ValueError("formal evaluation requires 239 unique FIPS")
        pairs = []
        with torch.inference_mode():
            for index in range(len(dataset)):
                images, weather, target_log, soil, fips, year = dataset[index]
                images = transform_eval_images(images.unsqueeze(0)).to(device)
                weather = weather.unsqueeze(0).to(device)
                soil = zscore_soil(soil.unsqueeze(0), mean, std).to(device)
                final_dist, _, _ = model(images, weather, soil)
                log_mean = float(final_dist[0, 0].cpu())
                pairs.append({
                    "FIPS": str(fips),
                    "year": int(year),
                    "yield_log": float(target_log),
                    "log_mean": log_mean,
                    "yield": math.exp(float(target_log)),
                    "prediction": math.exp(log_mean),
                })
        metrics = metrics_from_log_pairs(pairs)
        return {"metrics": metrics, "predictions": pairs, "checkpoint": str(checkpoint)}
    finally:
        dataset.close()


def main():
    args = build_parser().parse_args()
    result = evaluate(args)
    for path_arg, value in ((args.predictions_json, result["predictions"]),
                            (args.metrics_json, result["metrics"])):
        if path_arg:
            path = Path(path_arg)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
