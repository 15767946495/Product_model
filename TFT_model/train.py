"""
cropnet 训练脚本（参考 Product_model/Encoder_model/train.py 适配）。

监督微调协议：仅使用五州 2021 年有标签数据，固定训练 100 轮。
模型：TFTEncoderForYieldPrediction（来自 models.py，已移除 crop_phase 依赖）。

用法：
  python train.py
  python train.py --epochs 100 --lr 1e-3
  python train.py --use_crucial --hidden_size 32 --num_heads 4
"""

import argparse
import json
import os
import sys
import math
import random
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 项目内导入 ==========
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from models import TFTEncoderForYieldPrediction
from data import (
    load_jsonl,
    load_grid_cache,
    load_county_soil,
    compute_grid_global_stats,
    compute_soil_stats,
    build_grid_samples,
    GridTimeSeriesDataset,
    make_grid_collate_fn,
    AgricultureImageDataset,
    SOIL_DIM,
    DEFAULT_DYNAMIC_FEATURE_NAMES,
    DEFAULT_DATA_JSONL,
    DEFAULT_GRID_CACHE,
    DEFAULT_COUNTY_SOIL,
    GDD_FEATURE_NAME,
    CONSTRUCTED_FEATURES,
)
from cropnet_protocol import CROPNET_FIVE_STATES, PROTOCOL_MAX_STEPS, normalize_state_name
from experiments.protocol import DEFAULT_MODEL_IDENTITY

# ========== 常量 ==========
EARLY_STOP_PATIENCE: int = 2
FIXED_PROTOCOL_USES_VALIDATION = False
DEFAULT_TRAIN_YEAR = 2021
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_WARMUP_EPOCHS = 5
FINAL_CHECKPOINT_NAME = "final_model.pth"
DEFAULT_OUTPUT_DIR = "/data/raid0/hqx/TFT_train/fixed_2021"
DEFAULT_PRETRAINED_CHECKPOINT = str(
    Path(__file__).resolve().parents[1]
    / "MMST-ViT/output_dir/pvt_simclr/checkpoint-final.pth"
)
PRETRAIN_WEATHER_FEATURE_NAMES = (
    "Avg Temperature (K)", "Max Temperature (K)", "Min Temperature (K)",
    "Precipitation (kg m**-2)", "Relative Humidity (%)", "Wind Gust (m s**-1)",
    "Wind Speed (m s**-1)", "Downward Shortwave Radiation Flux (W m**-2)",
    "Vapor Pressure Deficit (kPa)",
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def last_valid_index(seq_lens: torch.Tensor) -> torch.Tensor:
    """返回每条序列的最后一个有效位置。"""
    return (seq_lens - 1).clamp_min(0)


# ============================================================
# 损失函数
# ============================================================
class PerSampleMSELoss(nn.Module):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, states=None) -> torch.Tensor:
        diff = inputs - targets
        return diff * diff


class BetaNLLLoss(nn.Module):
    """Beta-NLL for heteroscedastic regression in log-yield space."""

    def __init__(self, beta: float = 0.5, min_log_variance: float = -10.0,
                 max_log_variance: float = 5.0):
        super().__init__()
        self.beta = float(beta)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mean, log_variance = prediction.chunk(2, dim=-1)
        log_variance = log_variance.clamp(self.min_log_variance, self.max_log_variance)
        variance = torch.exp(log_variance)
        squared_error = (target - mean) ** 2
        # Standard beta-NLL uses the stop-gradient exponentiated variance
        # estimate as a per-sample weight; beta=0 recovers Gaussian NLL.
        weight = torch.exp(self.beta * log_variance.detach())
        loss = 0.5 * weight * (squared_error / variance + log_variance)
        if sample_valid is not None:
            sample_valid = sample_valid.to(device=loss.device, dtype=torch.bool)
            if sample_valid.shape != loss.shape[:1]:
                raise ValueError("sample_valid must have shape (B,)")
            loss = loss[sample_valid]
            if loss.numel() == 0:
                return prediction.sum() * 0.0
        return loss.mean()


def gaussian_temporal_kl(
    pred_dist: torch.Tensor,
    seq_lens: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    gamma: float = 0.05,
) -> torch.Tensor:
    """按显式月度有效掩码计算到最后有效月的高斯 KL。"""
    mean, log_var = pred_dist[..., 0], pred_dist[..., 1]
    B, T = mean.shape
    if valid_mask is None:
        if seq_lens is None:
            raise ValueError("seq_lens or valid_mask is required")
        valid_mask = torch.arange(T, device=mean.device)[None, :] < seq_lens[:, None]
    else:
        valid_mask = valid_mask.to(device=mean.device, dtype=torch.bool)
    last = valid_mask.long().cumsum(dim=1).argmax(dim=1)
    batch = torch.arange(B, device=mean.device)
    teacher_mean = mean[batch, last].detach()
    teacher_log_var = log_var[batch, last].detach().clamp(-10.0, 5.0)
    log_var = log_var.clamp(-10.0, 5.0)
    delta = last[:, None] - torch.arange(T, device=mean.device)[None, :]
    valid = valid_mask
    weights = torch.exp(-gamma * delta.to(mean.dtype)) * valid.to(mean.dtype)
    kl = 0.5 * (
        torch.exp(teacher_log_var[:, None] - log_var)
        + (mean - teacher_mean[:, None]) ** 2 * torch.exp(-log_var)
        - 1.0
        - (teacher_log_var[:, None] - log_var)
    )
    return (weights * kl).sum() / weights.sum().clamp_min(1.0)


def create_finetune_optimizer(model, lr: float, weight_decay: float):
    """Create the AdamW optimizer used by the MMST fine-tuning protocol."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def adjust_finetune_learning_rate(
    optimizer,
    progress: float,
    *,
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 0.0,
) -> float:
    """Set and return the MMST warmup/cosine LR at fractional epoch progress."""
    progress = max(0.0, min(float(progress), float(total_epochs)))
    if warmup_epochs > 0 and progress <= warmup_epochs:
        lr = base_lr * progress / warmup_epochs
    else:
        cosine_progress = (progress - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine_progress = max(0.0, min(cosine_progress, 1.0))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * cosine_progress))
    lr = max(float(min_lr), float(lr))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


class CRUCIALLoss(nn.Module):
    """CRUCIAL 损失（from reference train.py，精简版）。"""

    def __init__(self, lambda_reg=5e-5, ema_decay=0.9, sk_clamp=(-3.0, 4.0)):
        super().__init__()
        self.base_loss = PerSampleMSELoss()
        self.lambda_reg = lambda_reg
        self.ema_decay = ema_decay
        self.eps = 1e-7
        self.sk_clamp = sk_clamp
        self.mu_l = None
        self.sigma_l = None
        self.sk = None

    def lambertw(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=-1/math.e + self.eps)
        if hasattr(torch.special, "lambertw"):
            return torch.special.lambertw(x, k=0).real
        w = torch.zeros_like(x)
        mask_pos = x >= 0
        w[mask_pos] = torch.log(x[mask_pos] + 1.0)
        w[~mask_pos] = torch.clamp(x[~mask_pos] + 1.0, min=-2.0, max=2.0)
        for _ in range(5):
            exp_w = torch.exp(w)
            wew = w * exp_w
            w1e = (w + 1) * exp_w
            w = w - (wew - x) / (w1e + self.eps)
        return w

    def get_kappa(self, l: torch.Tensor, eps_t: torch.Tensor) -> torch.Tensor:
        beta = (l - eps_t) / (self.lambda_reg + self.eps)
        beta = torch.clamp(beta, min=-2/math.e)
        w = self.lambertw(beta / 2)
        kappa = torch.exp(-0.5 * w)
        return torch.clamp(kappa, min=self.eps)

    def forward(self, inputs, targets, sample_valid=None, states=None):
        l_raw = self.base_loss(inputs, targets, states)
        if l_raw.ndim > 1:
            l_raw = l_raw.reshape(l_raw.shape[0], -1).mean(dim=1)
        l_raw = l_raw.view(-1)
        if sample_valid is not None:
            sample_valid = sample_valid.to(device=l_raw.device, dtype=torch.bool)
            if sample_valid.shape != l_raw.shape:
                raise ValueError("sample_valid must have shape (B,)")
            l_raw = l_raw[sample_valid]
        return self._crucial_from_l_raw(l_raw, sample_valid=None)

    def _crucial_from_l_raw(
        self,
        l_raw: torch.Tensor,
        sample_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """对逐样本损失向量做 CRUCIAL 变换。"""
        if sample_valid is not None:
            sample_valid = sample_valid.to(device=l_raw.device, dtype=torch.bool)
            l_raw = l_raw[sample_valid]
        if l_raw.numel() == 0:
            return l_raw.sum() * 0.0
        with torch.no_grad():
            batch_mu = l_raw.mean()
            batch_sigma = l_raw.std(unbiased=False) + self.eps
            batch_diff = l_raw - batch_mu
            batch_sk = (batch_diff ** 3).mean() / (batch_sigma ** 3)
            batch_sk = torch.clamp(batch_sk, *self.sk_clamp)
            if self.mu_l is None:
                self.mu_l = batch_mu
                self.sigma_l = batch_sigma
                self.sk = batch_sk
            else:
                self.mu_l = self.ema_decay * self.mu_l + (1 - self.ema_decay) * batch_mu
                self.sigma_l = self.ema_decay * self.sigma_l + (1 - self.ema_decay) * batch_sigma
                self.sk = self.ema_decay * self.sk + (1 - self.ema_decay) * batch_sk
        eps_t = self.sk * self.mu_l
        kappa = self.get_kappa(l_raw, eps_t)
        loss_vec = kappa * (l_raw - eps_t) + self.lambda_reg * (torch.log(kappa) ** 2)
        return loss_vec.mean()


# ============================================================
# 验证指标
# ============================================================
def compute_rmse_r2(
    pred_raw: torch.Tensor, labels_raw: torch.Tensor, valid_mask: torch.Tensor
) -> Tuple[float, float]:
    """
    计算 RMSE 和 R²（在有效时间步上）。
    pred_raw, labels_raw: (B, T)，valid_mask: (B, T) bool
    返回 (rmse, r2)。
    """
    residual = pred_raw - labels_raw
    sq_err = residual * residual                         # (B, T)
    ss_res = sq_err[valid_mask].sum().item()             # 残差平方和
    n_valid = valid_mask.sum().item()
    if n_valid == 0:
        return float("nan"), float("nan")

    rmse = math.sqrt(ss_res / n_valid)

    # R²
    labels_flat = labels_raw[valid_mask]
    y_mean = labels_flat.mean().item()
    ss_tot = ((labels_flat - y_mean) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return rmse, r2


# ============================================================
# 训练流程
# ============================================================
def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    device: str = "cuda",
    ckpt_path: str = FINAL_CHECKPOINT_NAME,
    curve_path: str = "loss_curve.png",
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    use_crucial: bool = False,
    valid_freq: int = 1,
    use_rs: bool = False,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> float:
    """
    训练主循环。
    有验证集时保留旧的通用训练行为；固定协议路径传入 ``val_loader=None``，
    只训练指定轮数并在最后一轮保存最终模型。
    """
    model = model.to(device)
    _rc = model  # DataParallel wrapper for saving ckpt
    criterion = CRUCIALLoss() if use_crucial else BetaNLLLoss()
    optimizer = create_finetune_optimizer(model, lr, weight_decay)

    best_val_rmse = float("inf")
    remaining_patience = early_stop_patience

    train_loss_hist: List[float] = []
    val_rmse_hist: List[float] = []
    val_r2_hist: List[float] = []
    val_corr_hist: List[float] = []
    grad_modules = {
        "pvt": model.module.vit_encoder.backbone if isinstance(model, nn.DataParallel) else model.vit_encoder.backbone,
        "weather_grn": model.module.weather_context_grn if isinstance(model, nn.DataParallel) else model.weather_context_grn,
        "rs_grn": model.module.remote_context_grn if isinstance(model, nn.DataParallel) else model.remote_context_grn,
        "cross_modal": model.module.weather_remote_attn if isinstance(model, nn.DataParallel) else model.weather_remote_attn,
        "weather_vsn": model.module.grid_vsn if isinstance(model, nn.DataParallel) else model.grid_vsn,
        "spatial_cls": model.module.spatial_agg if isinstance(model, nn.DataParallel) else model.spatial_agg,
        "lstm": model.module.lstm_encoder if isinstance(model, nn.DataParallel) else model.lstm_encoder,
        "temporal_attn": model.module.attention if isinstance(model, nn.DataParallel) else model.attention,
        "mean_head": model.module.mean_head if isinstance(model, nn.DataParallel) else model.mean_head,
        "variance_head": model.module.variance_head if isinstance(model, nn.DataParallel) else model.variance_head,
    }

    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss_sum = 0.0
        train_samples_seen = 0
        grad_sums = {name: 0.0 for name in grad_modules}
        grad_counts = {name: 0 for name in grad_modules}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_index, batch in enumerate(pbar):
            adjust_finetune_learning_rate(
                optimizer,
                epoch + batch_index / max(len(train_loader), 1),
                base_lr=lr,
                warmup_epochs=warmup_epochs,
                total_epochs=epochs,
            )
            grid_feats, grid_coords, grid_mask, month_ids, day_ids, soil_feats, labels, seq_lens, *_rest = batch
            ag_images = _rest[4] if use_rs and len(_rest) >= 5 else None
            ag_mask = _rest[5] if use_rs and len(_rest) >= 6 else None
            ag_grid_mask = _rest[7] if use_rs and len(_rest) >= 8 else None

            grid_feats = grid_feats.to(device)
            grid_coords = grid_coords.to(device)
            grid_mask = grid_mask.to(device)
            month_ids = month_ids.to(device)
            day_ids = day_ids.to(device)
            soil_feats = soil_feats.to(device)
            labels = labels.to(device)
            seq_lens = seq_lens.to(device)
            if ag_images is not None:
                ag_images = ag_images.to(device)
            if ag_mask is not None:
                ag_mask = ag_mask.to(device)
            if ag_grid_mask is not None:
                ag_grid_mask = ag_grid_mask.to(device)

            pred, _, aux = model(
                    grid_feats=grid_feats,
                    grid_coords=grid_coords,
                    grid_mask=grid_mask,
                    soil_feats=soil_feats,
                    seq_lens=seq_lens,
                    month_ids=month_ids,
                    day_ids=day_ids,
                    ag_images=ag_images,
                    ag_mask=ag_mask,
                    ag_grid_mask=ag_grid_mask,
                )
            # 模型已直接返回每个县的最终预测 (B, 1)。
            sample_valid = aux["sample_valid"]
            log_labels = torch.log(labels.clamp_min(1e-6))
            if not bool(sample_valid.any()):
                continue

            if use_crucial:
                per_sample_mse = (pred[..., :1] - log_labels).square().squeeze(-1)
                loss = criterion._crucial_from_l_raw(per_sample_mse[sample_valid])
            else:
                loss = criterion(pred, log_labels, sample_valid) + 0.01 * gaussian_temporal_kl(
                    aux["grad_tensors"]["pred_dist_all"],
                    valid_mask=aux["month_valid"][sample_valid],
                )

            if not finetune_optimizer_step(loss, model, optimizer):
                continue

            for name, module in grad_modules.items():
                values = [
                    p.grad.detach().norm().item()
                    for p in module.parameters()
                    if p.grad is not None and torch.isfinite(p.grad).all()
                ]
                if values:
                    grad_sums[name] += math.sqrt(sum(value * value for value in values))
                    grad_counts[name] += 1
            batch_loss_sum, batch_samples_seen = accumulate_train_loss(loss, sample_valid)
            train_loss_sum += batch_loss_sum
            train_samples_seen += batch_samples_seen

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_sum / max(train_samples_seen, 1)
        print("  GradNorm | " + " ".join(
            f"{name}={grad_sums[name] / max(grad_counts[name], 1):.2e}"
            for name in grad_modules
        ))

        # 固定监督微调协议不读取验证集，也不使用验证指标选择 checkpoint。
        if val_loader is None:
            train_loss_hist.append(avg_train_loss)
            val_rmse_hist.append(float("nan"))
            val_r2_hist.append(float("nan"))
            val_corr_hist.append(float("nan"))
            continue

        # ========== 验证阶段 ==========
        if (epoch + 1) % valid_freq == 0:
            model.eval()
            all_pred_raw: List[float] = []
            all_label_raw: List[float] = []

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            with torch.no_grad():
                for batch in val_pbar:
                    grid_feats, grid_coords, grid_mask, month_ids, day_ids, soil_feats, labels, seq_lens, *_restv = batch
                    ag_images_v = _restv[4] if use_rs and len(_restv) >= 5 else None
                    ag_mask_v = _restv[5] if use_rs and len(_restv) >= 6 else None
                    ag_grid_mask_v = _restv[7] if use_rs and len(_restv) >= 8 else None
                    grid_feats = grid_feats.to(device)
                    grid_coords = grid_coords.to(device)
                    grid_mask = grid_mask.to(device)
                    month_ids = month_ids.to(device)
                    day_ids = day_ids.to(device)
                    soil_feats = soil_feats.to(device)
                    labels = labels.to(device)
                    seq_lens = seq_lens.to(device)
                    if ag_images_v is not None:
                        ag_images_v = ag_images_v.to(device)
                    if ag_mask_v is not None:
                        ag_mask_v = ag_mask_v.to(device)
                    if ag_grid_mask_v is not None:
                        ag_grid_mask_v = ag_grid_mask_v.to(device)

                    pred, _, _ = model(
                        grid_feats=grid_feats,
                        grid_coords=grid_coords,
                        grid_mask=grid_mask,
                        soil_feats=soil_feats,
                        seq_lens=seq_lens,
                        month_ids=month_ids,
                        day_ids=day_ids,
                        ag_images=ag_images_v,
                        ag_mask=ag_mask_v,
                        ag_grid_mask=ag_grid_mask_v,
                    )

                    # 模型输出为 log 单产，评估时还原到 bu/ac。
                    pred_raw = torch.exp(pred[..., 0])  # log-mean -> bu/ac
                    label_raw = labels.squeeze(-1)     # (B,)

                    all_pred_raw.extend(pred_raw.detach().cpu().tolist())
                    all_label_raw.extend(label_raw.detach().cpu().tolist())

                    batch_rmse = math.sqrt(
                            ((pred_raw - label_raw) ** 2).mean().item()
                    )
                    val_pbar.set_postfix({"rmse": f"{batch_rmse:.2f}"})

                if all_pred_raw:
                    pred_arr = np.array(all_pred_raw)
                    label_arr = np.array(all_label_raw)
                    resid = pred_arr - label_arr
                    ss_res = float((resid ** 2).sum())
                    n = len(pred_arr)
                    val_rmse = math.sqrt(ss_res / n)
                    ss_tot = float(((label_arr - label_arr.mean()) ** 2).sum())
                    val_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

                    # Pearson 相关系数
                    pred_centered = pred_arr - pred_arr.mean()
                    label_centered = label_arr - label_arr.mean()
                    cov = float((pred_centered * label_centered).sum())
                    std_p = float(np.sqrt((pred_centered ** 2).sum()))
                    std_l = float(np.sqrt((label_centered ** 2).sum()))
                    val_corr = cov / max(std_p * std_l, 1e-12)
                    val_corr = max(-1.0, min(1.0, val_corr))
                else:
                    val_rmse = float("nan")
                    val_r2 = float("nan")
                    val_corr = float("nan")

                train_loss_hist.append(avg_train_loss)
                val_rmse_hist.append(val_rmse)
                val_r2_hist.append(val_r2)
                val_corr_hist.append(val_corr)

                print(f" Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f} | R²: {val_r2:.4f} | Corr: {val_corr:.4f} | Best RMSE: {best_val_rmse:.4f}")

            # 绘制曲线
            _save_curves(curve_path, train_loss_hist, val_rmse_hist)

            # 早停
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                remaining_patience = early_stop_patience
                save_final_checkpoint(model, ckpt_path)
                print(f"  ✓ Val RMSE 创新低，已保存模型至 {ckpt_path}")
            else:
                remaining_patience -= 1
                print(f"  × Patience: {remaining_patience}/{early_stop_patience}")
                if remaining_patience <= 0:
                    print("  ■ 早停触发")
                    break
        else:
            train_loss_hist.append(avg_train_loss)
            val_rmse_hist.append(float("nan"))
            val_r2_hist.append(float("nan"))
            val_corr_hist.append(float("nan"))

    if val_loader is None:
        save_final_checkpoint(model, ckpt_path, epoch=epochs, optimizer=optimizer, **(checkpoint_metadata or {}))
        return train_loss_hist[-1] if train_loss_hist else float("nan")

    # 加载最佳 checkpoint
    if os.path.exists(ckpt_path):
        model.load_state_dict(load_checkpoint_state(ckpt_path))
    return best_val_rmse


def save_final_checkpoint(
    model, path: str, *, epoch: int = DEFAULT_EPOCHS, protocol=None,
    config=None, seed=None, feature_norm_path=None, hparams_path=None,
    optimizer=None,
    git_commit_value=None,
    manifest_hash=None, model_hparams=None, feature_norm=None,
    schedule=None, source_checkpoint=None,
) -> None:
    """保存固定协议最终权重，避免 DataParallel 的 module. 前缀泄漏。"""
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    config = dict(config or {})
    config["experiment_id"] = DEFAULT_MODEL_IDENTITY
    payload = {
        "model": state,
        "epoch": epoch,
        "protocol": protocol or {"finetune_year": DEFAULT_TRAIN_YEAR, "epochs": DEFAULT_EPOCHS},
        "config": config,
        "seed": seed,
        "feature_norm_path": feature_norm_path,
        "hparams_path": hparams_path,
        "experiment_id": DEFAULT_MODEL_IDENTITY,
        "git_commit": git_commit_value or git_commit(),
        "manifest_hash": manifest_hash,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "model_hparams": model_hparams or {},
        "feature_norm": feature_norm or {},
        "schedule": schedule,
        "source_checkpoint": source_checkpoint,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def build_checkpoint_metadata(
    *, protocol, config, seed, feature_norm_path, hparams_path,
    model_hparams, feature_norm, manifest_hash, schedule=None, source_checkpoint=None,
) -> Dict[str, Any]:
    """构造与 ``save_final_checkpoint`` 关键字签名一致的元数据。"""
    return {
        "protocol": protocol,
        "config": config,
        "seed": seed,
        "feature_norm_path": feature_norm_path,
        "hparams_path": hparams_path,
        "model_hparams": model_hparams,
        "feature_norm": feature_norm,
        "manifest_hash": manifest_hash,
        "schedule": schedule,
        "source_checkpoint": source_checkpoint,
    }


def load_pretrained_components(model, path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("model", payload) if isinstance(payload, dict) else payload
    target = model.module if isinstance(model, nn.DataParallel) else model
    target_state = target.state_dict()
    prefix_map = {
        "pvt.": "vit_encoder.backbone.",
        "vsn.": "grid_vsn.",
    }
    loaded, skipped = [], []
    mapped = {}
    for key, value in source.items():
        candidates = [key]
        for old, new in prefix_map.items():
            if key.startswith(old):
                candidates.append(new + key[len(old):])
        if key.startswith("vsn.flattened_grn."):
            skipped.append(key)
            continue
        if key.startswith("feature_projection."):
            suffix = key[len("feature_projection."):]
            index_text, _, parameter_suffix = suffix.partition(".")
            if index_text.isdigit():
                index = int(index_text)
                if index < len(PRETRAIN_WEATHER_FEATURE_NAMES):
                    name = PRETRAIN_WEATHER_FEATURE_NAMES[index]
                    if name in target.per_feature_linear:
                        candidates.append(f"per_feature_linear.{name}.{parameter_suffix}")
        if key.startswith("vsn.single_var_grns."):
            suffix = key[len("vsn.single_var_grns."):]
            index_text, _, parameter_suffix = suffix.partition(".")
            if index_text.isdigit() and int(index_text) < len(PRETRAIN_WEATHER_FEATURE_NAMES):
                name = PRETRAIN_WEATHER_FEATURE_NAMES[int(index_text)]
                if name in target.per_feature_linear:
                    candidates.append(f"grid_vsn.single_var_grns.{name}.{parameter_suffix}")
        found = next((candidate for candidate in candidates
                      if candidate in target_state and target_state[candidate].shape == value.shape), None)
        if found is None:
            skipped.append(key)
        else:
            mapped[found] = value
            loaded.append(found)
    target_state.update(mapped)
    target.load_state_dict(target_state, strict=False)
    return {
        "loaded": sorted(set(loaded)),
        "skipped": sorted(skipped),
        "manifest_hash": payload.get("manifest_hash") if isinstance(payload, dict) else None,
        "experiment_id": payload.get("experiment_id") if isinstance(payload, dict) else None,
    }


def load_mmst_pvt_backbone(model, checkpoint_path: str) -> Dict[str, Any]:
    """直接将官方 MMST checkpoint 的 PVT 参数加载到 TFT backbone。"""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(source, dict):
        raise ValueError("MMST checkpoint must contain a model state_dict")

    target = model.module if isinstance(model, nn.DataParallel) else model
    target_state = target.state_dict()
    loaded, skipped, mismatched = [], [], []
    mapped = {}

    for source_key, value in source.items():
        if not source_key.startswith("backbone."):
            skipped.append(source_key)
            continue

        suffix = source_key[len("backbone."):]
        target_suffix = None
        if suffix.startswith("patch_embed"):
            prefix, separator, remainder = suffix.partition(".")
            stage_text = prefix[len("patch_embed"):]
            if separator and stage_text.isdigit():
                target_suffix = f"stages.{int(stage_text) - 1}.{remainder}"
        elif suffix.startswith("block"):
            prefix, separator, remainder = suffix.partition(".")
            stage_text = prefix[len("block"):]
            block_text, separator2, parameter_suffix = remainder.partition(".")
            if separator and separator2 and stage_text.isdigit() and block_text.isdigit():
                target_suffix = (
                    f"blocks.{int(stage_text) - 1}.{block_text}.{parameter_suffix}"
                )
        elif suffix.startswith("pos_embed"):
            prefix, separator, remainder = suffix.partition(".")
            stage_text = prefix[len("pos_embed"):]
            if stage_text.isdigit():
                target_suffix = f"pos_embeds.{int(stage_text) - 1}"
                if separator:
                    target_suffix += f".{remainder}"
        elif suffix == "cls_token" or suffix.startswith("norm."):
            target_suffix = suffix

        if target_suffix is None:
            skipped.append(source_key)
            continue

        target_key = f"vit_encoder.backbone.{target_suffix}"
        if target_key not in target_state:
            skipped.append(source_key)
        elif target_state[target_key].shape != value.shape:
            mismatched.append(source_key)
        else:
            mapped[target_key] = value
            loaded.append(target_key)

    if not loaded:
        raise ValueError("MMST checkpoint contains no compatible TFT PVT tensors")

    missing_keys = []
    for stage in range(3):
        stage_prefixes = (
            f"vit_encoder.backbone.stages.{stage}.",
            f"vit_encoder.backbone.blocks.{stage}.",
        )
        required = [
            key for key in target_state
            if key.startswith(stage_prefixes)
        ]
        required.extend([f"vit_encoder.backbone.pos_embeds.{stage}"])
        missing_keys.extend(key for key in required if key not in mapped)
    if missing_keys:
        raise ValueError(
            "MMST checkpoint missing keys: " + ", ".join(sorted(set(missing_keys)))
        )

    target_state.update(mapped)
    target.load_state_dict(target_state, strict=False)
    return {
        "loaded": sorted(set(loaded)),
        "skipped": sorted(set(skipped)),
        "mismatched": sorted(set(mismatched)),
        "source_checkpoint": str(checkpoint_path),
        "source_protocol": payload.get("protocol") if isinstance(payload, dict) else None,
    }


def load_checkpoint_state(path: str):
    """加载裸 state_dict 或常见 model 包装，并去除 DataParallel 前缀。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a state_dict or model dict")
    return {
        key.removeprefix("module."): value
        for key, value in payload.items()
    }


def _save_curves(path: str, train_loss: List[float], val_rmse: List[float]) -> None:
    """保存训练损失和验证 RMSE 曲线。"""
    n = len(train_loss)
    if n == 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    xs = list(range(1, n + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, train_loss, label="Train Loss", color="#1f77b4")
    ax.set_ylabel("Loss", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax.twinx()
    ax2.plot(xs, val_rmse, label="Val RMSE", color="#d62728", linewidth=1.8)
    ax2.set_ylabel("RMSE", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 数据划分
# ============================================================
def split_train_val_by_year(
    samples: List[Dict], val_years: List[int]
) -> Tuple[List[Dict], List[Dict]]:
    """按 Year 划分：验证集 = Year ∈ val_years，训练集 = Year < min(val_years)。"""
    val_set = set(val_years)
    min_val = min(val_set)
    train, val = [], []
    for s in samples:
        y = int(s["Year"])
        if y in val_set:
            val.append(s)
        elif y < min_val:
            train.append(s)
    return train, val


def _split_pairs_by_year(
    pairs: List[Tuple[Dict, Dict]], val_years: List[int]
) -> Tuple[List[Tuple[Dict, Dict]], List[Tuple[Dict, Dict]]]:
    """按 Year 划分 (meta, entry) 对:验证集 = Year ∈ val_years,训练集 = Year < min(val_years)。"""
    val_set = set(val_years)
    min_val = min(val_set)
    train, val = [], []
    for m, e in pairs:
        y = int(m["Year"])
        if y in val_set:
            val.append((m, e))
        elif y < min_val:
            train.append((m, e))
    return train, val


def select_2021_training_pairs(
    pairs: List[Tuple[Dict, Dict]],
) -> List[Tuple[Dict, Dict]]:
    """选择固定协议的五州 2021 年有标签训练对，不涉及其他年份。"""
    selected = []
    for meta, entry in pairs:
        if not isinstance(meta, dict):
            raise ValueError("training pair metadata must be a dict")
        year = meta.get("Year", meta.get("year"))
        state = meta.get("State", meta.get("state"))
        if year is None or state is None:
            raise ValueError("training pair metadata requires Year/year and State/state")
        if int(year) == DEFAULT_TRAIN_YEAR and normalize_state_name(state) in CROPNET_FIVE_STATES:
            selected.append((meta, entry))
    selected, _ = filter_valid_label_pairs(selected)
    states = {
        normalize_state_name(meta.get("State", meta.get("state")))
        for meta, _ in selected
    }
    missing = CROPNET_FIVE_STATES - states
    if missing:
        raise ValueError(f"2021 training data missing protocol states: {sorted(missing)}")
    return selected


def final_checkpoint_name() -> str:
    return FINAL_CHECKPOINT_NAME


def validate_protocol_states(value: str):
    states = {normalize_state_name(item) for item in (value or "").split(",") if item.strip()}
    if states != CROPNET_FIVE_STATES:
        raise ValueError("training states must equal the complete CropNet five-state set")
    return states


def validate_protocol_args(
    epochs: int,
    train_year: int,
    early_stop: bool,
    checkpoint_name: str,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    pretrained: str,
) -> None:
    if epochs != DEFAULT_EPOCHS:
        raise ValueError("fixed protocol requires epochs=100")
    if train_year != DEFAULT_TRAIN_YEAR:
        raise ValueError("fixed protocol requires train_year=2021")
    if early_stop:
        raise ValueError("fixed protocol does not allow early stopping")
    if checkpoint_name != FINAL_CHECKPOINT_NAME:
        raise ValueError("fixed protocol checkpoint must be final_model.pth")
    if not math.isclose(lr, DEFAULT_LR, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("fixed protocol requires lr=1e-3")
    if not math.isclose(weight_decay, DEFAULT_WEIGHT_DECAY, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("fixed protocol requires weight_decay=0.05")
    if warmup_epochs != DEFAULT_WARMUP_EPOCHS:
        raise ValueError("fixed protocol requires warmup_epochs=5")
    if Path(pretrained).resolve() != Path(DEFAULT_PRETRAINED_CHECKPOINT).resolve():
        raise ValueError(
            "fixed protocol requires the canonical MMST PVT checkpoint: "
            f"{DEFAULT_PRETRAINED_CHECKPOINT}"
        )


def finetune_optimizer_step(loss: torch.Tensor, model, optimizer) -> bool:
    """Backpropagate one finite update and skip unsafe loss/gradient states."""
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(loss).all()):
        return False
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        optimizer.zero_grad(set_to_none=True)
        return False
    optimizer.step()
    return True


def accumulate_train_loss(
    loss: torch.Tensor, sample_valid: torch.Tensor
) -> Tuple[float, int]:
    """Accumulate a batch loss using only samples included in the loss."""
    valid_count = int(sample_valid.sum().item())
    return loss.item() * valid_count, valid_count


def filter_valid_label_pairs(pairs: List[Tuple[Dict, Dict]]) -> Tuple[List[Tuple[Dict, Dict]], int]:
    """过滤 meta 中 yield_per_acre 无效的 (meta, entry) 对。"""
    kept, dropped = [], 0
    for m, e in pairs:
        try:
            lab = m.get("yield_per_acre")
            if lab is None:
                dropped += 1
                continue
            v = float(lab)
            if not np.isfinite(v):
                dropped += 1
                continue
            kept.append((m, e))
        except (TypeError, ValueError):
            dropped += 1
    return kept, dropped


def filter_valid_label(samples: List[Dict]) -> Tuple[List[Dict], int]:
    """过滤目标字段 yield_per_acre 无效的样本。"""
    kept, dropped = [], 0
    for s in samples:
        try:
            lab = s.get("yield_per_acre")
            if lab is None:
                dropped += 1
                continue
            v = float(lab)
            if not np.isfinite(v):
                dropped += 1
                continue
            kept.append(s)
        except (TypeError, ValueError):
            dropped += 1
    return kept, dropped


def build_supervised_dataset(pairs, soil_dict, dynamic_feature_names, *, train, seed):
    samples = build_grid_samples(
        pairs, soil_dict=soil_dict, dynamic_feature_names=dynamic_feature_names
    )
    metadata = {
        (str(meta.get("FIPS", "")).zfill(5), int(meta.get("Year", meta.get("year")))): meta
        for meta, _ in pairs
    }
    ag_samples = []
    for sample in samples:
        meta = dict(metadata[(sample["FIPS"], sample["year"])])
        meta["State"] = normalize_state_name(meta.get("State", meta.get("state")))
        meta["Year"] = int(meta.get("Year", meta.get("year")))
        ag_samples.append(meta)
    ag_dataset = AgricultureImageDataset(ag_samples, train=train, seed=seed)
    return GridTimeSeriesDataset(samples, ag_dataset=ag_dataset)


# ============================================================
# 主入口
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="cropnet TFT 训练")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="随机采样的 batch 大小；PVT 全量训练默认使用 2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_year", type=int, default=DEFAULT_TRAIN_YEAR,
                        help="固定监督微调训练年份；协议要求为 2021")
    parser.add_argument("--states", type=str,
                        default=",".join(sorted(CROPNET_FIVE_STATES)),
                        help="按州过滤(逗号分隔的小写全称)，默认为协议五州")
    parser.add_argument("--grid_cache", type=str, default=None,
                        help="网格级气象缓存路径,默认 train_dataset/grid_cache.pt")
    parser.add_argument("--soil", type=str, default=None,
                        help="县级土壤数据路径,默认 soil_dataset/county_soil.json")
    parser.add_argument("--data_jsonl", type=str, default=None)
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="需能被 4 整除(时空位置编码 4 项一组)且能被 num_heads 整除")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup_epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--use_crucial", action="store_true",
                        help="使用 CRUCIAL 损失替代 MSE")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="产物输出目录，默认 /data/raid0/hqx/TFT_train/fixed_2021/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--early_stop", action="store_true",
                        help="保留旧接口但固定协议默认关闭；协议路径不启用早停")
    parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--checkpoint_name", type=str, default=FINAL_CHECKPOINT_NAME)
    parser.add_argument(
        "--pretrained",
        type=str,
        default=DEFAULT_PRETRAINED_CHECKPOINT,
        help="MMST PVT checkpoint",
    )
    parser.add_argument(
        "--use-constructed", dest="use_constructed",
        action=argparse.BooleanOptionalAction, default=True,
        help="固定协议默认启用全部农学构造特征；--no-use-constructed 仅用于消融",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    validate_protocol_args(
        args.epochs, args.train_year, args.early_stop, args.checkpoint_name,
        args.lr, args.weight_decay, args.warmup_epochs, args.pretrained,
    )

    if not os.path.isfile(args.pretrained):
        raise FileNotFoundError(
            "正式 TFT 微调需要 MMST PVT checkpoint，但未找到："
            f"{args.pretrained}。请先运行 MMST-ViT/run_pretrain.sh 生成该 checkpoint。"
        )

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # ========== 1. 加载数据(网格级) ==========
    jsonl_path = args.data_jsonl or DEFAULT_DATA_JSONL
    grid_cache_path = args.grid_cache or DEFAULT_GRID_CACHE
    print(f"\n[1] 加载数据:\n    jsonl: {jsonl_path}\n    网格缓存: {grid_cache_path}")
    meta_lines = load_jsonl(jsonl_path)
    cache = load_grid_cache(grid_cache_path)
    cache_entries = cache["entries"]
    if len(cache_entries) != len(meta_lines):
        raise ValueError(f"grid_cache 行数 {len(cache_entries)} 与 dataset.jsonl 行数 {len(meta_lines)} 不一致")
    print(f"  总样本数: {len(meta_lines)}")

    dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)
    if args.use_constructed:
        dynamic_feature_names += CONSTRUCTED_FEATURES    # 11+4=15 维农学构造特征
    else:
        dynamic_feature_names.append(GDD_FEATURE_NAME)   # 追加 CumGDD 通道(12 维,旧口径)
    pairs = list(zip(meta_lines, cache_entries))

    # 县级连续土壤静态特征
    soil_path = args.soil or DEFAULT_COUNTY_SOIL
    soil_dict = load_county_soil(soil_path)
    print(f"    县级土壤: {soil_path} ({len(soil_dict)} 县, 连续 {SOIL_DIM} 维, 不分桶)")

    use_rs = True
    train_ag = None
    val_ag = None
    permitted = validate_protocol_states(args.states)
    pairs = [
        p for p in pairs
        if normalize_state_name(p[0].get("State", p[0].get("state", ""))) in permitted
    ]
    print(f"  按州过滤 {sorted(permitted)} 后: {len(pairs)} 条")

    # 目标 = yield_per_acre(单产, bu/ac),无归一化
    train_pairs = select_2021_training_pairs(pairs)
    print(f"  Train (2021 only): {len(train_pairs)}")

    if not train_pairs:
        raise ValueError("2021 训练集为空，请检查数据")

    if use_rs:
        from data import AG_STATE_ABBR, AG_ROOT
        import h5py

        # 预扫描所有 AG h5 文件, 建立 (year,fips) 索引
        fips_has_ag = set()
        for h5_path in Path(AG_ROOT).rglob("*.h5"):
            year = next((int(x) for x in h5_path.parts if x.isdigit() and len(x) == 4), None)
            if year is None:
                continue
            try:
                with h5py.File(h5_path, "r") as f:
                    for fips_key in f.keys():
                        if len(str(fips_key)) == 5 and str(fips_key).isdigit():
                            fips_has_ag.add((year, str(fips_key)))
            except OSError:
                continue
        print(f"  AG h5 索引: {len(fips_has_ag)} 个 (year,fips) 条目")

        def _local_ag_ok(meta):
            try:
                fips = str(meta.get("FIPS", "")).zfill(5)
                year = int(meta.get("Year", meta.get("year")))
                return (year, fips) in fips_has_ag
            except Exception:
                return False

        n_tr = len(train_pairs)
        train_pairs_ag = [(m, e) for m, e in train_pairs if _local_ag_ok(m)]
        print(f"  AG 过滤 (本地 q2/q3 文件): Train {len(train_pairs_ag)}/{n_tr}")

        train_pairs = train_pairs_ag

        train_states = {
            normalize_state_name(meta.get("State", meta.get("state")))
            for meta, _ in train_pairs
        }
        missing_states = CROPNET_FIVE_STATES - train_states
        if missing_states:
            raise ValueError(
                f"2021 training data missing protocol states after filtering: "
                f"{sorted(missing_states)}"
            )

    # ========== 2. 目标 = 原始单产 (bu/ac),无归一化 ==========
    print("\n[2] 目标 = yield_per_acre(单产, bu/ac),不做归一化")

    # ========== 3. 特征标准化统计量(网格级气象 + 县级土壤,训练年份/训练县) ==========
    print("\n[3] 计算动态特征全局标准化统计量")
    global_stats = compute_grid_global_stats(train_pairs, dynamic_feature_names)
    # 土壤统计量只在训练县上算(与动态特征只在训练年份上算一致,避免验证县泄漏)
    train_fips = {str(m.get("FIPS", "")).zfill(5) for m, _ in train_pairs}
    soil_mean, soil_std = compute_soil_stats(soil_dict, fips_subset=train_fips)
    global_stats["soil"] = (soil_mean, soil_std)
    print(f"  动态特征: {len(dynamic_feature_names)} 维, 土壤: {SOIL_DIM} 维")
    print(f"  土壤统计量在 {len(train_fips)} 个训练县上计算")
    feature_norm_path = os.path.join(output_dir, "feature_norm.json")
    os.makedirs(os.path.dirname(feature_norm_path), exist_ok=True)
    payload = {}
    for k, (m, s) in global_stats.items():
        payload[k] = {"mean": m.detach().cpu().tolist(), "std": s.detach().cpu().tolist()}
    with open(feature_norm_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {feature_norm_path}")

    # ========== 4. 构建 DataLoader(网格级) ==========
    print("\n[4] 构建 DataLoader")
    collate_fn = make_grid_collate_fn(global_stats, dynamic_feature_names)

    train_dataset = build_supervised_dataset(
        train_pairs, soil_dict, dynamic_feature_names, train=True, seed=args.seed
    )

    # 随机采样,固定 batch_size,num_workers=0
    # (网格缓存 ~1.3GB,Windows 无 fork,多 worker 会整份 pickle 复制缓存)
    batch_size = args.batch_size
    print(f"  batch_size: {batch_size} (随机采样), Train: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"  Train batches: {len(train_loader)}; 不构建 2022 验证集")

    # ========== 5. 创建模型 ==========
    print("\n[5] 创建模型")
    model = TFTEncoderForYieldPrediction(
        soil_dim=SOIL_DIM,
        dynamic_feature_names=dynamic_feature_names,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        output_size=1,
        num_heads=args.num_heads,
    )

    transfer_report = load_mmst_pvt_backbone(model, args.pretrained)
    print(f"  预训练迁移: loaded={len(transfer_report['loaded'])}, skipped={len(transfer_report['skipped'])}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数总量: {total_params:,}, 可训练: {trainable_params:,}")

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel: {torch.cuda.device_count()} GPUs, batch_size={args.batch_size}")
    print(f"  构造特征: use_constructed={args.use_constructed}  "
          f"网格级天气-遥感联合注意力")

    # 保存模型超参
    hparams = {
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "num_lstm_layers": args.num_lstm_layers,
        "dropout": args.dropout,
        "dynamic_feature_dim": len(dynamic_feature_names),
        "soil_dim": SOIL_DIM,
        "loss": "crucial" if args.use_crucial else "beta_nll",
        "use_constructed": bool(args.use_constructed),
        "pretrained": args.pretrained,
        "transfer_report": transfer_report,
    }
    with open(os.path.join(output_dir, "model_hparams.json"), "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)

    # ========== 6. 训练 ==========
    print("\n[6] 开始训练")
    print(f"  损失: {'CRUCIAL' if args.use_crucial else 'Beta-NLL + temporal KL'}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Weight Decay: {args.weight_decay}, Warmup: {args.warmup_epochs}")
    print(f"  固定协议: {args.epochs} epochs, 无早停, 无验证/最佳 checkpoint 选择")

    ckpt_path = os.path.join(output_dir, args.checkpoint_name)
    curve_path = os.path.join(output_dir, "loss_curve.png")

    final_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        ckpt_path=ckpt_path,
        curve_path=curve_path,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        early_stop_patience=args.early_stop_patience,
        use_crucial=args.use_crucial,
        use_rs=use_rs,
        checkpoint_metadata=build_checkpoint_metadata(
            protocol={"finetune_year": args.train_year, "epochs": args.epochs, "checkpoint_rule": "final_epoch"},
            config=vars(args),
            seed=args.seed,
            feature_norm_path=feature_norm_path,
            hparams_path=os.path.join(output_dir, "model_hparams.json"),
            model_hparams=hparams,
            feature_norm=payload,
            manifest_hash=transfer_report.get("manifest_hash"),
            schedule={
                "name": "warmup_cosine",
                "base_lr": args.lr,
                "min_lr": 0.0,
                "warmup_epochs": args.warmup_epochs,
                "total_epochs": args.epochs,
                "optimizer": "AdamW",
                "betas": [0.9, 0.999],
                "weight_decay": args.weight_decay,
            },
             source_checkpoint=args.pretrained,
        ),
    )

    print(f"\n{'='*50}")
    print(f"训练完成！第 {args.epochs} 轮 Train Loss: {final_loss:.6f}")
    print(f"模型: {ckpt_path}")
    print(f"曲线: {curve_path}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
