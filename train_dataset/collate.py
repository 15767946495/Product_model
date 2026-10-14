"""统一批处理：变长网格 padding + mask；GNN 五年窗口图批处理。"""

from __future__ import annotations

from typing import Sequence

import torch

from train_dataset.dataset import DAYS_PER_MONTH, REMOTE_MONTHS


def collate_county_year(batch: Sequence[dict]) -> dict:
    """weather (B,G,168,11)、coords (B,G,2)、grid_mask (B,G)、soil (B,7)、labels (B,)；
    long_term (B,G,4,12,11) + long_term_mask (B,G)；有图像时 images (B,6,G,3,S,S) + image_mask (B,6,G)。

    padding 网格 mask 为 False；真实网格即使数值全零 mask 仍为 True。
    """
    if not batch:
        raise ValueError("empty batch")
    sizes = [item["weather"].shape[0] for item in batch]
    B, G = len(batch), max(sizes)
    T, F = batch[0]["weather"].shape[1:]
    weather = torch.zeros((B, G, T, F), dtype=torch.float32)
    coords = torch.zeros((B, G, 2), dtype=torch.float32)
    grid_mask = torch.zeros((B, G), dtype=torch.bool)
    long_term = torch.zeros((B, G, 4, 12, F), dtype=torch.float32)
    long_term_mask = torch.zeros((B, G), dtype=torch.bool)
    for b, (item, n) in enumerate(zip(batch, sizes)):
        weather[b, :n] = item["weather"]
        coords[b, :n] = item["coords"]
        grid_mask[b, :n] = True
        if item["long_term"] is not None:
            long_term[b, :n] = item["long_term"]
            long_term_mask[b, :n] = True
    out = {
        "weather": weather, "coords": coords, "grid_mask": grid_mask,
        "soil": torch.stack([item["soil"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "long_term": long_term, "long_term_mask": long_term_mask,
        "sample_id": [item["sample_id"] for item in batch],
        "fips": [item["fips"] for item in batch],
        "year": torch.tensor([item["year"] for item in batch]),
        "grid_ids": [item["grid_ids"] for item in batch],
    }
    has_images = ["images" in item for item in batch]
    if any(has_images):
        if not all(has_images):
            raise ValueError("batch mixes items with and without images")
        M, _, C, H, W = batch[0]["images"].shape
        images = torch.zeros((B, M, G, C, H, W), dtype=torch.float32)
        image_mask = torch.zeros((B, M, G), dtype=torch.bool)
        for b, (item, n) in enumerate(zip(batch, sizes)):
            images[b, :, :n] = item["images"]
            image_mask[b, :, :n] = True
        out["images"], out["image_mask"] = images, image_mask
    return out


def monthly_weather_view(weather: torch.Tensor) -> torch.Tensor:
    """(B,G,168,11) -> (B,6,G,28,11)；只在批处理时 reshape，不改写 JSONL。"""
    B, G, T, F = weather.shape
    months = len(REMOTE_MONTHS)
    if T != months * DAYS_PER_MONTH:
        raise ValueError(f"expected {months * DAYS_PER_MONTH} days, got {T}")
    return weather.reshape(B, G, months, DAYS_PER_MONTH, F).permute(0, 2, 1, 3, 4).contiguous()


def collate_gnn_window(batch: Sequence[Sequence[dict]]) -> dict:
    """batch: 每县一个五年窗口（Dataset item 列表）。

    weather (N,5,168,11) 为县内有效网格平均；soil (N,7)；years (5,)；targets (N,) 为窗口最后一年标签；
    edge_index (2,E) 来自 neighbor_fips 并添加自环。标签不进入节点特征。
    """
    if not batch:
        raise ValueError("empty batch")
    windows = sorted(batch, key=lambda w: w[0]["fips"])
    years = [item["year"] for item in windows[0]]
    if len(years) != 5:
        raise ValueError(f"GNN window must have 5 years, got {years}")
    fips = []
    for window in windows:
        if [item["year"] for item in window] != years:
            raise ValueError(f"{window[0]['fips']}: window years differ from {years}")
        if len({item["fips"] for item in window}) != 1:
            raise ValueError("window mixes counties")
        if len({tuple(item["neighbor_fips"]) for item in window}) != 1:
            raise ValueError(f"{window[0]['fips']}: neighbor_fips differs across window")
        fips.append(window[0]["fips"])
    if len(set(fips)) != len(fips):
        raise ValueError("duplicate county in GNN batch")
    position = {f: i for i, f in enumerate(fips)}
    weather = torch.stack([torch.stack([item["weather"].mean(dim=0) for item in w]) for w in windows])
    soil = torch.stack([w[0]["soil"] for w in windows])
    targets = torch.stack([w[-1]["label"] for w in windows])
    edges = {(i, i) for i in range(len(fips))}
    for i, window in enumerate(windows):
        for n in window[0]["neighbor_fips"]:
            if n in position:
                edges.add((i, position[n]))
                edges.add((position[n], i))
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return {
        "weather": weather, "soil": soil, "years": torch.tensor(years), "targets": targets,
        "fips": fips, "edge_index": edge_index,
        "sample_id": [[item["sample_id"] for item in w] for w in windows],
    }
