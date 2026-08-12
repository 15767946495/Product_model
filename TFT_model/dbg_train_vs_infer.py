#!/usr/bin/env python3
"""诊断: train loop 报告的 loss(RMSE~47) vs infer(19.76) 差异来源。
用与 train.py 完全相同的流水线构建 train loader(构造特征 15 维),eval 模式下算:
  A) 训练循环式: 逐样本取末个有效步(=8月且非padding)  -> RMSE
  B) infer 式:    全局最大有效步索引(季末)           -> RMSE
并打印 seq_len / 末步索引分布,判断差异是否来自"短序列样本末步落在季中"。
"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

from data import (
    load_jsonl, load_grid_cache, load_county_soil,
    DEFAULT_DATA_JSONL, DEFAULT_GRID_CACHE, DEFAULT_COUNTY_SOIL,
    DEFAULT_DYNAMIC_FEATURE_NAMES, CONSTRUCTED_FEATURES, SOIL_DIM,
    compute_grid_global_stats, compute_soil_stats, build_grid_samples,
    make_grid_collate_fn, GridTimeSeriesDataset,
)
from train import _split_pairs_by_year, filter_valid_label_pairs
from models import TFTEncoderForYieldPrediction

CKPT = r"train_output/ablation2d_val2021_hs16_h1_lstm1_c15/mean/best_model.pth"

def main():
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # ---- 与 train.py 完全一致的数据流水线 ----
    meta_lines = load_jsonl(DEFAULT_DATA_JSONL)
    cache = load_grid_cache(DEFAULT_GRID_CACHE)
    cache_entries = cache["entries"]
    pairs = list(zip(meta_lines, cache_entries))
    state_set = {"minnesota","wisconsin","michigan","iowa","illinois","indiana","ohio","missouri","kentucky"}
    pairs = [p for p in pairs if str(p[0].get("State","")).lower() in state_set]
    soil_dict = load_county_soil(DEFAULT_COUNTY_SOIL)

    train_pairs, val_pairs = _split_pairs_by_year(pairs, [2021])
    train_pairs, _ = filter_valid_label_pairs(train_pairs)
    print(f"train_pairs={len(train_pairs)}")

    dyn_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES) + list(CONSTRUCTED_FEATURES)
    global_stats = compute_grid_global_stats(train_pairs, dyn_names)
    train_fips = {str(m.get("FIPS","")).zfill(5) for m,_ in train_pairs}
    soil_mean, soil_std = compute_soil_stats(soil_dict, fips_subset=train_fips)
    global_stats["soil"] = (soil_mean, soil_std)

    collate_fn = make_grid_collate_fn(global_stats, dyn_names)
    train_samples = build_grid_samples(train_pairs, soil_dict=soil_dict, dynamic_feature_names=dyn_names)
    train_ds = GridTimeSeriesDataset(train_samples)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"train_ds={len(train_ds)}, batches={len(train_loader)}")

    # ---- 模型: 与 checkpoint 一致的 hparams ----
    model = TFTEncoderForYieldPrediction(
        soil_dim=SOIL_DIM, dynamic_feature_names=dyn_names,
        hidden_size=16, num_lstm_layers=1, dropout=0.2, output_size=1,
        num_heads=1, spatial_mode="mean", use_grid_rope=False, use_time_rope=False,
    )
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model = model.to(device).eval()

    # ---- 模型 dropout 分布 ----
    def collect_dropout(m, prefix=""):
        for name, child in m.named_children():
            if isinstance(child, torch.nn.Dropout):
                print(f"  Dropout(p={child.p}) @ {prefix}{name}")
            collect_dropout(child, prefix + name + ".")
    print("模型 dropout 层:")
    collect_dropout(model)

    # 收集
    per_sample_pred, per_sample_label, per_sample_lastidx, per_sample_seqlens = [], [], [], []
    global_preds, global_labels = [], []   # 全局最大有效步
    seq_lens_all = []

    Tmax = None
    for bi, batch in enumerate(train_loader):
        grid_feats, grid_coords, grid_mask, month_ids, soil_feats, labels, seq_lens, states, years, _, _ = batch
        grid_feats=grid_feats.to(device); grid_coords=grid_coords.to(device); grid_mask=grid_mask.to(device)
        month_ids=month_ids.to(device); soil_feats=soil_feats.to(device); seq_lens=seq_lens.to(device)
        pred_all, _, _ = model(grid_feats=grid_feats, grid_coords=grid_coords, grid_mask=grid_mask,
                               soil_feats=soil_feats, seq_lens=seq_lens)
        B, T, _ = pred_all.shape
        if Tmax is None: Tmax = T
        pred_all = pred_all.squeeze(-1)   # (B,T)
        labels = labels.squeeze(-1)       # (B,)
        seq_len_np = seq_lens.cpu().numpy()
        seq_lens_all.extend(seq_len_np.tolist())

        pad_mask = torch.arange(T, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        month_mask = month_ids >= 8
        valid_mask = pad_mask & month_mask
        time_idx = torch.arange(T, device=device).unsqueeze(0)
        last_valid_idx = torch.where(valid_mask, time_idx, -1).max(dim=1).values.clamp(min=0)

        for b in range(B):
            idx = int(last_valid_idx[b].item())
            p = pred_all[b, idx].item()
            l = labels[b].item()
            per_sample_pred.append(p); per_sample_label.append(l)
            per_sample_lastidx.append(idx); per_sample_seqlens.append(int(seq_len_np[b]))

        # 全局最大有效步: 所有样本中有效步索引的最大值
        g = int(valid_mask.float().argmax(dim=1).max().item()) if valid_mask.any() else T-1
        global_preds.extend(pred_all[:, g].cpu().tolist())
        global_labels.extend(labels.cpu().tolist())

    def rmse(p, l):
        p=np.array(p); l=np.array(l)
        return float(math.sqrt(((p-l)**2).mean()))

    a = rmse(per_sample_pred, per_sample_label)
    b = rmse(global_preds, global_labels)
    print(f"\n[A] 逐样本末步 (train.py 损失口径) RMSE = {a:.4f}   (n={len(per_sample_pred)})")
    print(f"[B] 全局最大有效步 (infer 口径)    RMSE = {b:.4f}   (n={len(global_preds)})")

    s = np.array(per_sample_seqlens)
    li = np.array(per_sample_lastidx)
    print(f"\nseq_len: min={s.min()} p25={np.percentile(s,25):.0f} med={np.median(s):.0f} "
          f"p75={np.percentile(s,75):.0f} max={s.max()}")
    print(f"末步索引: min={li.min()} p25={np.percentile(li,25):.0f} med={np.median(li):.0f} "
          f"p75={np.percentile(li,75):.0f} max={li.max()}")
    print(f"末步索引==max({li.max()}) 的样本数: {(li==li.max()).sum()} / {len(li)}")
    # 早季末步(索引 < max-30,即提前>=31天)的样本占比
    early = (li < li.max()-30).sum()
    print(f"末步比全局末步提前>=31天: {early} / {len(li)}")

    # 末步很早(<季末45天)样本的 RMSE vs 其余
    late_mask = li >= li.max()-15
    a_late = rmse([p for p, m in zip(per_sample_pred, late_mask) if m],
                  [l for l, m in zip(per_sample_label, late_mask) if m])
    a_early = rmse([p for p, m in zip(per_sample_pred, late_mask) if not m],
                   [l for l, m in zip(per_sample_label, late_mask) if not m])
    print(f"\n末步在季末窗口(<=15天前)内: n={(late_mask).sum()}  RMSE={a_late:.4f}")
    print(f"末步提前>15天            : n={(~late_mask).sum()}  RMSE={a_early:.4f}")

    # ===== 同一批数据, train 模式(dropout ON) vs eval 模式, 对比 loss =====
    model.train()
    t_preds, t_labels = [], []
    n_batch = 0
    for bi, batch in enumerate(train_loader):
        grid_feats, grid_coords, grid_mask, month_ids, soil_feats, labels, seq_lens, states, years, _, _ = batch
        grid_feats=grid_feats.to(device); grid_coords=grid_coords.to(device); grid_mask=grid_mask.to(device)
        month_ids=month_ids.to(device); soil_feats=soil_feats.to(device); seq_lens=seq_lens.to(device)
        pred_all, _, _ = model(grid_feats=grid_feats, grid_coords=grid_coords, grid_mask=grid_mask,
                               soil_feats=soil_feats, seq_lens=seq_lens)
        B, T, _ = pred_all.shape
        pred_all = pred_all.squeeze(-1)
        labels = labels.squeeze(-1)
        pad_mask = torch.arange(T, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        month_mask = month_ids >= 8
        valid_mask = pad_mask & month_mask
        time_idx = torch.arange(T, device=device).unsqueeze(0)
        last_valid_idx = torch.where(valid_mask, time_idx, -1).max(dim=1).values.clamp(min=0)
        for b in range(B):
            idx = int(last_valid_idx[b].item())
            t_preds.append(pred_all[b, idx].item())
            t_labels.append(labels[b].item())
        n_batch += 1
        if n_batch >= 50:
            break
    model.eval()
    a_train = rmse(t_preds, t_labels)
    print(f"\n[train 模式(dropout ON), 前{n_batch}批] 逐样本末步 RMSE = {a_train:.4f}  (n={len(t_preds)})")

    # 用 eval 模式对同一前 50 批重算
    e_preds, e_labels = [], []
    for bi, batch in enumerate(train_loader):
        if bi >= n_batch: break
        grid_feats, grid_coords, grid_mask, month_ids, soil_feats, labels, seq_lens, states, years, _, _ = batch
        grid_feats=grid_feats.to(device); grid_coords=grid_coords.to(device); grid_mask=grid_mask.to(device)
        month_ids=month_ids.to(device); soil_feats=soil_feats.to(device); seq_lens=seq_lens.to(device)
        pred_all, _, _ = model(grid_feats=grid_feats, grid_coords=grid_coords, grid_mask=grid_mask,
                               soil_feats=soil_feats, seq_lens=seq_lens)
        B, T, _ = pred_all.shape
        pred_all = pred_all.squeeze(-1)
        labels = labels.squeeze(-1)
        pad_mask = torch.arange(T, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        month_mask = month_ids >= 8
        valid_mask = pad_mask & month_mask
        time_idx = torch.arange(T, device=device).unsqueeze(0)
        last_valid_idx = torch.where(valid_mask, time_idx, -1).max(dim=1).values.clamp(min=0)
        for b in range(B):
            idx = int(last_valid_idx[b].item())
            e_preds.append(pred_all[b, idx].item())
            e_labels.append(labels[b].item())
    a_eval = rmse(e_preds, e_labels)
    print(f"[eval  模式(dropout OFF), 同{n_batch}批] 逐样本末步 RMSE = {a_eval:.4f}  (n={len(e_preds)})")

if __name__ == "__main__":
    main()
