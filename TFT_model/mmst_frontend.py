"""Exact official MMST spatial front-end for TFT remote-sensing features."""

from __future__ import annotations

import importlib.util
import sys
from threading import RLock
from pathlib import Path

import torch
from torch import nn


_MMST_ROOT = Path(__file__).resolve().parents[1] / "MMST-ViT"
_OFFICIAL_LOAD_LOCK = RLock()
_OFFICIAL_CLASS_CACHE = None


def _load_official_module(name: str, relative_path: str, *, aliases=()):
    path = _MMST_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载官方 MMST-ViT 模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    for alias in aliases:
        sys.modules.setdefault(alias, module)
    return module


def _load_official_frontend_classes():
    global _OFFICIAL_CLASS_CACHE
    with _OFFICIAL_LOAD_LOCK:
        if _OFFICIAL_CLASS_CACHE is not None:
            return _OFFICIAL_CLASS_CACHE

        result = _load_official_frontend_classes_locked()
        _OFFICIAL_CLASS_CACHE = result
        return result


def _load_official_frontend_classes_locked():
    saved = {
        name: sys.modules.get(name)
        for name in ("models_pvt", "models_pvt_simclr", "attention")
    }
    try:
        attention = _load_official_module(
            "_tft_mmst_official_attention_frontend", "attention.py"
        )
        sys.modules["attention"] = attention
        models_pvt = _load_official_module(
            "_tft_mmst_official_models_pvt_frontend", "models_pvt.py"
        )
        sys.modules["models_pvt"] = models_pvt
        simclr = _load_official_module(
            "_tft_mmst_official_models_pvt_simclr_frontend",
            "models_pvt_simclr.py",
        )
        sys.modules["models_pvt_simclr"] = simclr
        mmst = _load_official_module(
            "_tft_mmst_official_models_mmst_vit_frontend", "models_mmst_vit.py"
        )
        return attention, simclr, mmst
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


(
    _OFFICIAL_ATTENTION_MODULE,
    _OFFICIAL_SIMCLR_MODULE,
    _OFFICIAL_MMST_MODULE,
) = _load_official_frontend_classes()
PVTSimCLR = _OFFICIAL_SIMCLR_MODULE.PVTSimCLR
SpatialTransformer = _OFFICIAL_ATTENTION_MODULE.SpatialTransformer
MMST_ViT = _OFFICIAL_MMST_MODULE.MMST_ViT


def _checkpoint_state(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("MMST checkpoint must contain a state dictionary")
    state = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise ValueError("MMST checkpoint model state must be a dictionary")
    return state


def _normalize_checkpoint_state(state):
    prefixed = {key.startswith("module.") for key in state}
    normalized = {}
    duplicates = []
    for key, value in state.items():
        name = key.removeprefix("module.")
        if name in normalized:
            duplicates.append(name)
        normalized[name] = value
    if len(prefixed) > 1:
        names = ", ".join(sorted(set(duplicates)))
        suffix = f"; duplicate normalized keys: {names}" if names else ""
        raise ValueError(f"checkpoint has mixed module. and unprefixed keys{suffix}")
    if duplicates:
        names = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"duplicate normalized checkpoint keys: {names}")
    return normalized


def _audit_extra_keys(keys):
    spatial = []
    unknown = []
    for key in keys:
        if key in {"space_token", "pos_embedding"} or key.startswith("space_transformer."):
            spatial.append(key)
        else:
            unknown.append(key)
    return sorted(spatial), sorted(unknown)


class MMSTSpatialFrontend(nn.Module):
    """Return official MMST spatial CLS features for six monthly image grids.

    ``images`` must already be transformed CHW tensors with shape
    ``(B, 6, G, 3, 224, 224)``. Transform selection belongs to the training
    entry point, which may use the official ``DataWrapper``.
    """

    def __init__(self, checkpoint_path, dim=512, num_grid=64, batch_size=64):
        super().__init__()
        self.dim = dim
        self.num_grid = num_grid
        self.batch_size = batch_size
        self.pvt_backbone = PVTSimCLR(
            "pvt_tiny", out_dim=dim, context_dim=9, pretrained=False
        )
        checkpoint_state = _checkpoint_state(checkpoint_path)
        expected_keys = set(self.pvt_backbone.state_dict())
        checkpoint_state = _normalize_checkpoint_state(checkpoint_state)
        missing = expected_keys - set(checkpoint_state)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise RuntimeError(f"missing PVTSimCLR checkpoint keys: {missing_names}")
        self.pvt_backbone.load_state_dict(
            {name: checkpoint_state[name] for name in expected_keys}, strict=True
        )
        for parameter in self.pvt_backbone.parameters():
            parameter.requires_grad = False
        self.pvt_backbone.eval()
        self.loaded_pvt_keys = sorted(expected_keys)
        self.ignored_spatial_keys, self.ignored_unknown_keys = _audit_extra_keys(
            set(checkpoint_state) - expected_keys
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 6, num_grid, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = SpatialTransformer(
            dim, 4, 3, 64, mult=4, dropout=0.0
        )
        self.dropout = nn.Dropout(0.0)

    def train(self, mode=True):
        super().train(mode)
        self.pvt_backbone.eval()
        return self

    def forward_features(self, images, short_weather):
        batch, months, grids, channels, height, width = images.shape
        flat_images = images.reshape(batch * months * grids, channels, height, width)
        flat_weather = short_weather.reshape(batch * months * grids, *short_weather.shape[3:])
        count = flat_images.shape[0]
        chunks = []
        for start in range(0, count, self.batch_size):
            image_chunk = flat_images[start:start + self.batch_size]
            weather_chunk = flat_weather[start:start + self.batch_size]
            if image_chunk.shape[0] == 1:
                image_chunk = torch.cat((image_chunk, image_chunk), dim=0)
                weather_chunk = torch.cat((weather_chunk, weather_chunk), dim=0)
                chunks.append(self.pvt_backbone(
                    image_chunk, context=weather_chunk,
                )[:1])
            else:
                chunks.append(self.pvt_backbone(
                    image_chunk, context=weather_chunk,
                ))
        return torch.cat(chunks, dim=0)

    def forward(self, images, short_weather):
        if images.ndim != 6 or images.shape[1] != 6:
            raise ValueError("images must have shape (B, 6, G, 3, 224, 224)")
        if images.shape[3:] != (3, 224, 224):
            raise ValueError("images must be CHW tensors with shape (3, 224, 224)")
        if short_weather.ndim != 5 or tuple(short_weather.shape) != (
            images.shape[0], 6, images.shape[2], 28, 9
        ):
            raise ValueError("short_weather must have shape (B, 6, G, 28, 9)")
        if images.shape[2] > self.num_grid - 1:
            raise ValueError(
                f"number of grids G must satisfy G <= num_grid={self.num_grid} - 1 "
                f"({self.num_grid - 1})"
            )

        batch, months, grids = images.shape[:3]
        features = self.forward_features(images, short_weather)
        features = features.reshape(batch, months, grids, self.dim)
        cls_space_tokens = self.space_token.expand(batch, months, -1, -1)
        features = torch.cat((cls_space_tokens, features), dim=2)
        features = features + self.pos_embedding[:, :, :grids + 1]
        features = self.dropout(features)
        features = features.reshape(batch * months, grids + 1, self.dim)
        features = self.space_transformer(features)
        return features[:, 0].reshape(batch, months, self.dim)
