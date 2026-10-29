"""骨干构造、SpectralGPT+ 初始化（与上游 train_da.py / train_ft.py 相同的加载逻辑）与加载报告。"""

from __future__ import annotations

from pathlib import Path

import torch

import config as cfg
from model import models_vit_tensor
from model.models_PhenoYieldNet import ViT_multimodal_multicrop
from train_dataset.run_output import sha256_file
from util.pos_embed import interpolate_pos_embed

# 统一数据时序轴只有 4--9 月（索引 0 = 4 月）；上游各作物窗口与该轴取交集后全部为 0--5。
GROWTH_CYCLES_APR_SEP = {crop: list(range(cfg.MONTHS)) for crop in range(cfg.NUM_CROPS)}


def build_backbone(drop_path: float, num_classes: int) -> torch.nn.Module:
    return models_vit_tensor.__dict__[cfg.BACKBONE](drop_path_rate=drop_path, num_classes=num_classes)


def load_pretrained_state(net: torch.nn.Module, state_dict: dict) -> dict:
    """按上游规则适配并加载已读取的 state dict。"""
    checkpoint_model = dict(state_dict)
    state_dict = net.state_dict()
    removed = []
    for k in ['patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            removed.append(k)
            del checkpoint_model[k]
    interpolate_pos_embed(net, checkpoint_model)
    msg = net.load_state_dict(checkpoint_model, strict=False)
    return {
        "removed_shape_mismatch": removed,
        "loaded_keys": sorted(set(checkpoint_model) & set(state_dict)),
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
    }


def load_pretrained(net: torch.nn.Module, path, *, expected_sha256: str | None = None) -> dict:
    """读取 checkpoint，并按上游规则适配形状、插值位置编码和加载权重。"""
    path = Path(path)
    report = {"path": str(path)}
    if expected_sha256 is not None:
        digest = sha256_file(path)
        if digest != expected_sha256:
            raise ValueError(f"{path}: sha256 {digest} != expected {expected_sha256}")
        report["sha256"] = digest
    checkpoint_model = torch.load(path, map_location="cpu", weights_only=False)["model"]
    report.update(load_pretrained_state(net, checkpoint_model))
    return report


def build_model(model_config: dict, backbone: torch.nn.Module) -> ViT_multimodal_multicrop:
    return ViT_multimodal_multicrop(
        num_short_term_seq=cfg.MONTHS, out_dim=1, num_grid=cfg.NUM_GRID, backbone=backbone, context_dim=9,
        dim=cfg.EMBED_DIM, batch_size=model_config["batch_size"], num_crops=cfg.NUM_CROPS,
        grad_checkpointing=model_config["grad_checkpointing"], freeze_backbone=model_config["frozen_fm"],
        crop_growth_cycles=GROWTH_CYCLES_APR_SEP,
    )
