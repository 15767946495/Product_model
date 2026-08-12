"""
cropnet 训练脚本（参考 Product_model/Encoder_model/train.py 适配）。

默认验证年：2022（训练集使用 2022 年之前的数据）。
模型：TFTEncoderForYieldPrediction（来自 models.py，已移除 crop_phase 依赖）。

用法：
  python train.py
  python train.py --epochs 500 --lr 1e-3 --val_year 2022
  python train.py --use_crucial --hidden_size 32 --num_heads 4
"""

import argparse
import json
import os
import sys
import math
import random
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
    SOIL_DIM,
    DEFAULT_DYNAMIC_FEATURE_NAMES,
    DEFAULT_DATA_JSONL,
    DEFAULT_GRID_CACHE,
    DEFAULT_COUNTY_SOIL,
    GDD_FEATURE_NAME,
    CONSTRUCTED_FEATURES,
)

# ========== 常量 ==========
EARLY_STOP_PATIENCE: int = 10


# ============================================================
# 损失函数
# ============================================================
class PerSampleMSELoss(nn.Module):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, states=None) -> torch.Tensor:
        diff = inputs - targets
        return diff * diff


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

    def forward(self, inputs, targets, states=None):
        l_raw = self.base_loss(inputs, targets, states).view(-1)
        return self._crucial_from_l_raw(l_raw)

    def _crucial_from_l_raw(self, l_raw: torch.Tensor) -> torch.Tensor:
        """对逐样本损失向量做 CRUCIAL 变换。"""
        with torch.no_grad():
            batch_mu = l_raw.mean()
            batch_sigma = l_raw.std() + self.eps
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
    val_loader,
    epochs: int = 500,
    lr: float = 5e-4,
    device: str = "cuda",
    ckpt_path: str = "best_model.pth",
    curve_path: str = "loss_curve.png",
    weight_decay: float = 5e-4,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    use_crucial: bool = False,
    valid_freq: int = 1,
) -> float:
    """
    训练主循环。
    返回最佳验证 RMSE。
    """
    model = model.to(device)
    criterion = CRUCIALLoss() if use_crucial else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_rmse = float("inf")
    remaining_patience = early_stop_patience

    train_loss_hist: List[float] = []
    val_rmse_hist: List[float] = []
    val_r2_hist: List[float] = []
    val_corr_hist: List[float] = []

    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss_sum = 0.0
        train_samples_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            # batch: (grid_feats, grid_coords, grid_mask, month, static_bucket_ids, labels, seq_lens, states, years, fips, counties)
            grid_feats, grid_coords, grid_mask, month_ids, soil_feats, labels, seq_lens, states, years, _, _ = batch
            grid_feats = grid_feats.to(device)
            grid_coords = grid_coords.to(device)
            grid_mask = grid_mask.to(device)
            month_ids = month_ids.to(device)
            soil_feats = soil_feats.to(device)
            labels = labels.to(device)
            seq_lens = seq_lens.to(device)

            optimizer.zero_grad()

            pred_all, _, _ = model(
                grid_feats=grid_feats,
                grid_coords=grid_coords,
                grid_mask=grid_mask,
                soil_feats=soil_feats,
                seq_lens=seq_lens,
            )

            # === 损失：只取 >=8 月的时间步 ===
            B, T, _ = pred_all.shape
            pad_mask = torch.arange(T, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)  # (B, T)
            month_mask = month_ids >= 8                                    # (B, T)
            both_mask = pad_mask & month_mask                              # (B, T)

            # 每个样本只取最后一步有效步（>=8月且非padding）计算损失
            has_valid = both_mask.any(dim=1)                              # (B,)
            time_idx = torch.arange(T, device=device).unsqueeze(0)        # (1, T)
            last_valid_idx = (time_idx * both_mask).argmax(dim=1)         # (B,)
            pred_last = pred_all[torch.arange(B, device=device), last_valid_idx]  # (B, 1)
            per_sample_mse = (pred_last - labels) ** 2                    # (B, 1)
            per_sample_mse = per_sample_mse * has_valid.unsqueeze(-1).float()  # 无效样本 loss = 0

            if use_crucial:
                loss = criterion._crucial_from_l_raw(per_sample_mse.view(-1))
            else:
                loss = per_sample_mse.mean()

            if torch.isnan(loss):
                continue

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_samples_seen += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_sum / max(train_samples_seen, 1)

        # ========== 验证阶段 ==========
        if (epoch + 1) % valid_freq == 0:
            model.eval()
            all_pred_raw: List[float] = []
            all_label_raw: List[float] = []

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            with torch.no_grad():
                for batch in val_pbar:
                    grid_feats, grid_coords, grid_mask, month_ids, soil_feats, labels, seq_lens, states, years, _, _ = batch
                    grid_feats = grid_feats.to(device)
                    grid_coords = grid_coords.to(device)
                    grid_mask = grid_mask.to(device)
                    month_ids = month_ids.to(device)
                    soil_feats = soil_feats.to(device)
                    labels = labels.to(device)
                    seq_lens = seq_lens.to(device)

                    pred_all, _, _ = model(
                        grid_feats=grid_feats,
                        grid_coords=grid_coords,
                        grid_mask=grid_mask,
                        soil_feats=soil_feats,
                        seq_lens=seq_lens,
                    )

                    # 在每个样本 >=8 月的有效步中，取最后一步作为输出
                    B, T, _ = pred_all.shape
                    pad_mask = torch.arange(T, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
                    month_mask = month_ids >= 8
                    valid_mask = pad_mask & month_mask                     # (B, T)
                    time_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
                    last_valid_idx = torch.where(valid_mask, time_idx, -1).max(dim=1).values  # (B,)
                    last_valid_idx = last_valid_idx.clamp(min=0)
                    batch_idx = torch.arange(B, device=device)
                    pred_last = pred_all[batch_idx, last_valid_idx]        # (B, 1)

                    # 无归一化,预测/标签已在原始单产空间 (bu/ac),直接使用
                    pred_raw = pred_last.squeeze(-1)   # (B,)
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
                torch.save(model.state_dict(), ckpt_path)
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

    # 加载最佳 checkpoint
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return best_val_rmse


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


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="cropnet TFT 训练")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="随机采样的 batch 大小(9 州玉米带部分县 G≤134,取 8 防显存溢出)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_year", type=str, default="2022",
                        help="验证年份，可用逗号分隔多个年份，如 '2021,2022'")
    parser.add_argument("--states", type=str,
                        default="minnesota,wisconsin,michigan,iowa,illinois,indiana,ohio,missouri,kentucky",
                        help="按州过滤(逗号分隔的小写全称)。默认 DeepCropNet 9 玉米带州")
    parser.add_argument("--grid_cache", type=str, default=None,
                        help="网格级气象缓存路径,默认 train_dataset/grid_cache.pt")
    parser.add_argument("--soil", type=str, default=None,
                        help="县级土壤数据路径,默认 soil_dataset/county_soil.json")
    parser.add_argument("--data_jsonl", type=str, default=None)
    parser.add_argument("--hidden_size", type=int, default=32,
                        help="需能被 4 整除(时空位置编码 4 项一组)且能被 num_heads 整除")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--use_crucial", action="store_true",
                        help="使用 CRUCIAL 损失替代 MSE")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="产物输出目录，默认 train_model/train_output/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--spatial_mode", type=str, default="attention",
                        choices=["attention", "mean"],
                        help="网格→县聚合方式: attention(空间注意力,默认) / mean(直接网格均值,消融对照)")
    parser.add_argument("--use_gdd", action="store_true",
                        help="追加累计积温 CumGDD 通道(base 8°C,由 Avg Temperature 按网格累计)")
    parser.add_argument("--use_constructed", action="store_true",
                        help="追加全部农学构造特征(CumGDD/KDD/CumPRCP/CumDeficit,15 维动态输入)"
                             ";与 --use_gdd 互斥,开启时以本开关为准")
    parser.add_argument("--grid_rope", action="store_true",
                        help="网格注意力用 2D RoPE(lat/lng,仅 Q/K,CLS 位置=0 空间中性,不含时间),替换加性正余弦时空位置编码")
    parser.add_argument("--time_rope", action="store_true",
                        help="时序因果注意力的 Q/K 加一维时间 RoPE")
    args = parser.parse_args()

    # 解析验证年份
    val_years = [int(y.strip()) for y in args.val_year.split(",") if y.strip()]
    if not val_years:
        val_years = [2022]

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 输出目录
    val_tag = "_".join(str(y) for y in val_years)
    output_dir = args.output_dir or os.path.join(_THIS_DIR, "train_output", f"val_{val_tag}")
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
    elif args.use_gdd:
        dynamic_feature_names.append(GDD_FEATURE_NAME)   # 追加 CumGDD 通道(12 维,旧口径)
    pairs = list(zip(meta_lines, cache_entries))

    # 县级连续土壤静态特征
    soil_path = args.soil or DEFAULT_COUNTY_SOIL
    soil_dict = load_county_soil(soil_path)
    print(f"    县级土壤: {soil_path} ({len(soil_dict)} 县, 连续 {SOIL_DIM} 维, 不分桶)")

    # 可选的州过滤(对齐论文 MMST-ViT 的 4 州设置:MS/LA/IA/IL)
    if args.states:
        state_set = {s.strip().lower() for s in args.states.split(",") if s.strip()}
        pairs = [p for p in pairs if str(p[0].get("State", "")).lower() in state_set]
        print(f"  按州过滤 {sorted(state_set)} 后: {len(pairs)} 条")

    # 目标 = yield_per_acre(单产, bu/ac),无归一化
    train_pairs, val_pairs = _split_pairs_by_year(pairs, val_years)
    train_pairs, dropped_train = filter_valid_label_pairs(train_pairs)
    val_pairs, dropped_val = filter_valid_label_pairs(val_pairs)
    print(f"  Train: {len(train_pairs)} (dropped {dropped_train}), Val: {len(val_pairs)} (dropped {dropped_val})")

    if not train_pairs or not val_pairs:
        raise ValueError("训练集或验证集为空，请检查 --val_year 和数据")

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

    train_samples = build_grid_samples(train_pairs, soil_dict=soil_dict, dynamic_feature_names=dynamic_feature_names)
    val_samples = build_grid_samples(val_pairs, soil_dict=soil_dict, dynamic_feature_names=dynamic_feature_names)
    train_dataset = GridTimeSeriesDataset(train_samples)
    val_dataset = GridTimeSeriesDataset(val_samples)

    # 随机采样,固定 batch_size,num_workers=0
    # (网格缓存 ~1.3GB,Windows 无 fork,多 worker 会整份 pickle 复制缓存)
    batch_size = args.batch_size
    print(f"  batch_size: {batch_size} (随机采样), Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

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
        spatial_mode=args.spatial_mode,
        use_grid_rope=args.grid_rope,
        use_time_rope=args.time_rope,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数总量: {total_params:,}, 可训练: {trainable_params:,}")
    print(f"  消融开关: use_constructed={args.use_constructed}  use_gdd={args.use_gdd}  "
          f"grid_rope={args.grid_rope}  time_rope={args.time_rope}")

    # 保存模型超参
    hparams = {
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "num_lstm_layers": args.num_lstm_layers,
        "dropout": args.dropout,
        "dynamic_feature_dim": len(dynamic_feature_names),
        "soil_dim": SOIL_DIM,
        "loss": "crucial" if args.use_crucial else "mse",
        "spatial_mode": args.spatial_mode,
        "use_gdd": bool(args.use_gdd),
        "use_constructed": bool(args.use_constructed),
        "grid_rope": bool(args.grid_rope),
        "time_rope": bool(args.time_rope),
    }
    with open(os.path.join(output_dir, "model_hparams.json"), "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)

    # ========== 6. 训练 ==========
    print("\n[6] 开始训练")
    print(f"  损失: {'CRUCIAL' if args.use_crucial else 'MSE'}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Weight Decay: {args.weight_decay}")
    print(f"  早停 patience: {args.early_stop_patience}")

    ckpt_path = os.path.join(output_dir, "best_model.pth")
    curve_path = os.path.join(output_dir, "loss_curve.png")

    best_rmse = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        ckpt_path=ckpt_path,
        curve_path=curve_path,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        use_crucial=args.use_crucial,
    )

    print(f"\n{'='*50}")
    print(f"训练完成！最佳 Val RMSE: {best_rmse:.6f}")
    print(f"模型: {ckpt_path}")
    print(f"曲线: {curve_path}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
