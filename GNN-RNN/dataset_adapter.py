"""GNN-RNN 唯一数据适配边界：统一 CropNetDataset -> 两个五年县图窗口。

训练窗口 2017--2021 预测 2021；测试窗口 2018--2022 预测 2022（train_dataset.dataset.GNN_WINDOWS）。
标准化统计量只来自训练窗口：天气按 (县, 年, 天) 求每变量均值/方差，土壤按县求，
标签只用 2021 训练目标。2022 标签只作为最终指标的 y_true，不进入任何输入或统计量。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from graph import check_same_graph, validate_graph  # noqa: E402
from train_dataset.collate import collate_gnn_window  # noqa: E402
from train_dataset.dataset import DATA_ROOT, GNN_WINDOWS, YEARS, CropNetDataset, build_gnn_windows  # noqa: E402

TRAIN_YEARS = GNN_WINDOWS["train"]
TEST_YEARS = GNN_WINDOWS["test"]
TARGET_TRAIN_YEAR = TRAIN_YEARS[-1]
TARGET_TEST_YEAR = TEST_YEARS[-1]
EPS = 1e-6


def load_windows(jsonl_path, data_root=DATA_ROOT) -> dict:
    """返回 {"train": batch, "test": batch}；两窗口县集合、节点顺序、边已校验一致且只来自 Census 接壤。"""
    dataset = CropNetDataset(Path(jsonl_path), YEARS, load_images=False, data_root=Path(data_root))
    items = [dataset[i] for i in range(len(dataset))]
    windows = build_gnn_windows(items)
    batches = {split: collate_gnn_window(windows[split]) for split in ("train", "test")}
    check_same_graph(batches["train"], batches["test"])
    neighbors = {w[0]["fips"]: w[0]["neighbor_fips"] for w in windows["train"]}
    for batch in batches.values():
        validate_graph(batch["fips"], batch["edge_index"], neighbors)
    for split, years in GNN_WINDOWS.items():
        if batches[split]["years"].tolist() != list(years):
            raise ValueError(f"{split} window years {batches[split]['years'].tolist()} != {list(years)}")
    # 输出顺序与统一 Dataset 中目标年份样本顺序一致（按 fips 排序）
    batches["test"]["target_sample_ids"] = [ids[-1] for ids in batches["test"]["sample_id"]]
    batches["train"]["target_sample_ids"] = [ids[-1] for ids in batches["train"]["sample_id"]]
    return batches


def fit_stats(train: dict) -> dict:
    """只在训练窗口上计算标准化统计量。"""
    weather = train["weather"]  # (N,5,168,11)
    soil = train["soil"]        # (N,7)
    targets = train["targets"]  # (N,) 2021
    return {
        "weather_mean": weather.mean(dim=(0, 1, 2)), "weather_std": weather.std(dim=(0, 1, 2)).clamp_min(EPS),
        "soil_mean": soil.mean(dim=0), "soil_std": soil.std(dim=0).clamp_min(EPS),
        "y_mean": targets.mean(), "y_std": targets.std().clamp_min(EPS),
    }


def model_inputs(batch: dict, stats: dict) -> dict:
    """标准化后的模型输入；不含任何标签。"""
    return {
        "weather": (batch["weather"] - stats["weather_mean"]) / stats["weather_std"],
        "soil": (batch["soil"] - stats["soil_mean"]) / stats["soil_std"],
        "edge_index": batch["edge_index"],
    }


def stats_to_json(stats: dict) -> dict:
    return {k: v.tolist() for k, v in stats.items()}


def stats_from_json(payload: dict) -> dict:
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in payload.items()}
