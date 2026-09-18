"""
RS Cross-Attention 融合路径诊断。

检查遥感分支各阶段输出：
  1. ViT 编码输出统计
  2. RS 空间注意力后 rs_vec 统计
  3. 气象窗口 met_windows 统计
  4. Cross-Attn 输出 fuse 统计
  5. fuse vs rs_vec 差异（Cross-Attn 是否改变了 RS 向量）
  6. temporal_feat 注入前后的变化（注入是否被保留）

用法:
  python tests/diagnose_rs_fusion.py
"""

import json, math, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
_TFT_DIR = _PROJ_ROOT / "TFT_model"
sys.path.insert(0, str(_TFT_DIR))
sys.path.insert(0, str(_PROJ_ROOT))

from models import TFTEncoderForYieldPrediction
from data import (
    load_jsonl, load_grid_cache, load_county_soil,
    build_grid_samples, GridTimeSeriesDataset, make_grid_collate_fn,
    SOIL_DIM, DEFAULT_DYNAMIC_FEATURE_NAMES,
    DEFAULT_DATA_JSONL, DEFAULT_GRID_CACHE, DEFAULT_COUNTY_SOIL,
)

_stats = {}

def _hook_rs_encoded(module, inp, out):
    """Hook ViT 的 proj 输出"""
    _stats["rs_encoded"] = {
        "mean": out.detach().mean().item(),
        "std": out.detach().std().item(),
        "norm": out.detach().norm().item(),
        "norm_per_sample": out.detach().norm(dim=-1).mean().item(),
    }

def _capture_forward_remote_sensing(model, batch, device):
    """捕获 _forward_remote_sensing 内部各阶段的值。"""
    grid_feats, grid_coords, grid_mask, _, _, soil_feats, labels, seq_lens, *_rest = batch
    ag_images = _rest[4].to(device) if len(_rest) >= 5 and _rest[4] is not None else None

    grid_feats = grid_feats.to(device)
    grid_coords = grid_coords.to(device)
    grid_mask = grid_mask.to(device)
    soil_feats = soil_feats.to(device)
    seq_lens = seq_lens.to(device)

    B, G, T, _ = grid_feats.shape
    H = model.hidden_size

    # --- 先跑一次气象路径得到 temporal_feat（不含遥感） ---
    with torch.no_grad():
        _, _, aux_no_rs = model(
            grid_feats=grid_feats, grid_coords=grid_coords,
            grid_mask=grid_mask, soil_feats=soil_feats,
            seq_lens=seq_lens, ag_images=None,
        )
    tf_no_rs = aux_no_rs["grad_tensors"]["temporal_feat"].clone()

    # --- 完整模型跑一次 ---
    with torch.no_grad():
        pred_all, attn_t, aux_full = model(
            grid_feats=grid_feats, grid_coords=grid_coords,
            grid_mask=grid_mask, soil_feats=soil_feats,
            seq_lens=seq_lens, ag_images=ag_images,
        )
    tf_with_rs = aux_full["grad_tensors"]["temporal_feat"].clone()

    # --- 手动逐步执行 RS 路径获取中间值 ---
    _N_rs = 12
    A = ag_images
    imgs = A.permute(0, 2, 1, 3, 4, 5).reshape(B * G, _N_rs, 3, 224, 224)
    imgs = imgs.reshape(B * G * _N_rs, 3, 224, 224)

    with torch.no_grad():
        # Step 1: ViT
        rs_enc = model.vit_encoder(imgs)
        rs_enc = rs_enc.reshape(B, G, _N_rs, H)
        rs_enc = rs_enc * grid_mask[:, :, None, None].to(rs_enc.dtype)

        # Step 2: RS spatial attention
        rs_tok = rs_enc.permute(0, 1, 2, 3).contiguous()
        rs_tok = rs_tok * grid_mask[:, :, None, None].to(rs_tok.dtype)
        rs_vec, rs_sp_w = model.rs_spatial_agg.forward_weights(
            rs_tok, grid_coords, grid_mask)

        # Step 3: Build met_windows
        from models import _rs_day_indices, _build_met_windows, RS_WINDOW_SIZE
        day_idx = _rs_day_indices(device)
        met_w, wm = _build_met_windows(tf_no_rs, seq_lens, day_idx, RS_WINDOW_SIZE)

        # Step 4: Cross-Attention
        fuse = model.rs_met_cross_attn(rs_vec, met_w, wm)

    # --- 统计报告 ---
    print("=" * 60)
    print("RS 融合路径逐阶段诊断 (第 1 个 batch, B={}, G={})".format(B, G))
    print("=" * 60)

    print(f"\n[1] ViT 编码输出 (DINOv2-small → GRN proj):")
    print(f"    形状: {rs_enc.shape}  (B,G,12,H)")
    print(f"    均值: {rs_enc.mean().item():.4f}  标准差: {rs_enc.std().item():.4f}")
    print(f"    有效值范围: [{rs_enc.min().item():.4f}, {rs_enc.max().item():.4f}]")
    print(f"    零值比例: {(rs_enc.abs() < 1e-6).float().mean().item():.1%}")
    # 检查各网格的 ViT 输出是否都一样
    enc_var_across_grids = rs_enc.std(dim=1).mean().item()
    print(f"    跨网格标准差均值: {enc_var_across_grids:.4f}  "
          f"{'⚠ 网格间差异小' if enc_var_across_grids < 0.05 else '✓'}")
    # 检查各时相的 ViT 输出是否都一样
    enc_var_across_time = rs_enc.std(dim=2).mean().item()
    print(f"    跨时相标准差均值: {enc_var_across_time:.4f}  "
          f"{'⚠ 时相间差异小' if enc_var_across_time < 0.05 else '✓'}")

    print(f"\n[2] RS 空间注意力输出 rs_vec:")
    print(f"    形状: {rs_vec.shape}  (B,12,H)")
    print(f"    均值: {rs_vec.mean().item():.4f}  标准差: {rs_vec.std().item():.4f}")
    rs_vec_var_t = rs_vec.std(dim=1).mean().item()
    print(f"    跨时相标准差: {rs_vec_var_t:.4f}  "
          f"{'⚠ 12个时相几乎一样' if rs_vec_var_t < 0.05 else '✓ 时相有差异'}")
    # RS 跨样本的 cosine 相似度
    if B >= 2:
        cos_sim = nn.functional.cosine_similarity(
            rs_vec[0].mean(0).unsqueeze(0), rs_vec[1].mean(0).unsqueeze(0)).item()
        print(f"    样本间余弦相似度: {cos_sim:.4f}  "
              f"{'⚠ 不同县RS表示几乎相同!' if cos_sim > 0.95 else '✓ 有区分度'}")

    print(f"\n[3] 气象窗口 met_windows:")
    print(f"    形状: {met_w.shape}  (B,12,14,H)")
    n_empty = (~wm.any(dim=-1)).float().mean().item()
    print(f"    空窗口比例: {n_empty:.1%}")
    print(f"    窗口内均值: {met_w.mean().item():.4f}  标准差: {met_w.std().item():.4f}")

    print(f"\n[4] Cross-Attention 输出 fuse:")
    print(f"    形状: {fuse.shape}  (B,12,H)")
    print(f"    均值: {fuse.mean().item():.4f}  标准差: {fuse.std().item():.4f}")
    fuse_var_t = fuse.std(dim=1).mean().item()
    print(f"    跨时相标准差: {fuse_var_t:.4f}")

    # fuse vs rs_vec 是否相同
    diff_fuse_rs = (fuse - rs_vec).abs().mean().item()
    cos_fuse_rs = nn.functional.cosine_similarity(
        fuse.reshape(-1, H), rs_vec.reshape(-1, H), dim=-1).mean().item()
    print(f"    fuse - rs_vec MAE: {diff_fuse_rs:.4f}")
    print(f"    cosine_sim(fuse, rs_vec): {cos_fuse_rs:.4f}  "
          f"{'⚠ fuse ≈ rs_vec, Cross-Attn 无效!' if cos_fuse_rs > 0.98 else '✓ Cross-Attn 有效改变'}")

    # Cross-Attention 输出与气象窗口的关系
    met_mean = met_w.mean(dim=2)  # (B,12,H)
    cos_fuse_met = nn.functional.cosine_similarity(
        fuse.reshape(-1, H), met_mean.reshape(-1, H), dim=-1).mean().item()
    print(f"    cosine_sim(fuse, met_mean): {cos_fuse_met:.4f}  "
          f"{'⚠ fuse ≈ met_mean, RS信息被覆盖' if cos_fuse_met > 0.98 else 'OK'}")

    print(f"\n[5] temporal_feat 注入分析:")
    diff_tf = (tf_with_rs - tf_no_rs).abs()
    n_injected = (diff_tf.max(dim=-1).values > 1e-6).float().mean().item()
    max_change = diff_tf.max().item()
    mean_change = diff_tf[diff_tf > 1e-6].mean().item() if diff_tf.max() > 1e-6 else 0
    print(f"    temporal_feat 有变化的元素比例: {n_injected:.2%}")
    print(f"    最大变化: {max_change:.4f}")
    print(f"    非零变化的均值: {mean_change:.4f}")
    if n_injected < 0.01:
        print(f"    ⚠ temporal_feat 几乎未被修改 — 注入失败或Cross-Attn输出≈原值")

    # RS 相关的时间步注入效果
    day_idx_clamped = _rs_day_indices(device).clamp(max=T - 1)
    for i_rs in range(0, _N_rs, 3):  # 抽样几个 RS 时相
        tidx = int(day_idx_clamped[i_rs].item())
        if tidx < T:
            diff_at_idx = diff_tf[0, tidx].abs().mean().item()
            print(f"    时相{i_rs:2d} (t={tidx:3d}): "
                  f"temporal_feat 变化 = {diff_at_idx:.4f}")

    # --- 结论 ---
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)
    issues = []

    if enc_var_across_grids < 0.05:
        issues.append("ViT: 不同网格的编码几乎相同 → ViT 没捕捉到网格间差异")
    if enc_var_across_time < 0.05:
        issues.append("ViT: 不同时相的编码几乎相同 → 时相信息丢失")
    if cos_fuse_rs > 0.98:
        issues.append("Cross-Attn: fuse ≈ rs_vec → Cross-Attention 未产生有效变换")
    if cos_fuse_met > 0.98:
        issues.append("Cross-Attn: fuse ≈ met_mean → RS 信息被气象均值覆盖")
    if n_injected < 0.05:
        issues.append("注入: temporal_feat 几乎未被修改 → RS 路径整体无效")

    if not issues:
        issues.append("无明显瓶颈，RS 路径各环节均正常 — 检查是否训练不足或学习率过小")

    for i, iss in enumerate(issues):
        print(f"  [{i+1}] {iss}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/data/raid0/hqx/TFT_train/val_2022")
    parser.add_argument("--val_year", type=str, default="2022")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # Load hparams
    hp_path = os.path.join(args.output_dir, "model_hparams.json")
    with open(hp_path, "r") as f:
        hp = json.load(f)
    hidden_size = int(hp["hidden_size"])
    num_heads = int(hp["num_heads"])
    num_lstm_layers = int(hp["num_lstm_layers"])
    dropout = float(hp["dropout"])
    spatial_mode = str(hp["spatial_mode"])
    use_rs = bool(hp.get("use_remote_sensing", False))
    print(f"[模型] hidden={hidden_size} heads={num_heads} sp={spatial_mode} rs={use_rs}")

    # Load norm
    norm_path = os.path.join(args.output_dir, "feature_norm.json")
    with open(norm_path, "r") as f:
        fn = json.load(f)
    global_stats = {}
    for k, v in fn.items():
        global_stats[k] = (torch.tensor(v["mean"], dtype=torch.float32),
                            torch.tensor(v["std"], dtype=torch.float32))

    # Load data (same as infer.py)
    jsonl_path, grid_cache_path = DEFAULT_DATA_JSONL, DEFAULT_GRID_CACHE
    meta_lines = load_jsonl(jsonl_path)
    cache = load_grid_cache(grid_cache_path)
    if len(cache["entries"]) != len(meta_lines):
        raise ValueError("行数不一致")

    from cropnet_protocol import CROPNET_FIVE_STATES
    pairs = list(zip(meta_lines, cache["entries"]))
    pairs = [p for p in pairs
             if str(p[0].get("State", "")).lower() in CROPNET_FIVE_STATES]
    val_years = [int(y) for y in args.val_year.split(",")]
    val_set = set(val_years)
    val_pairs = [(m, e) for m, e in pairs if int(m["Year"]) in val_set]
    kept = []
    for m, e in val_pairs:
        try:
            if np.isfinite(float(m.get("yield_per_acre", np.nan))):
                kept.append((m, e))
        except (TypeError, ValueError):
            pass
    val_pairs = kept

    dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)
    soil_dict = load_county_soil(DEFAULT_COUNTY_SOIL)

    from data import (AG_STATE_ABBR, load_ag_manifest, DEFAULT_AG_MANIFEST,
                      AgricultureImageDataset)
    raw_ag = load_ag_manifest(DEFAULT_AG_MANIFEST)
    ag_available = set()
    for (abbr, yr, fips_str), entries in raw_ag.items():
        qs = {e["path"].rstrip(".h5").split("_")[-1][-5:] for e in entries}
        if "06-30" in qs and "09-30" in qs:
            ag_available.add((abbr, int(yr), str(fips_str).zfill(5)))

    val_pairs_ag = []
    for m, e in val_pairs:
        abbr = AG_STATE_ABBR.get(str(m.get("State", "")).strip().lower())
        if abbr is None:
            continue
        if (abbr, int(m["Year"]), str(m.get("FIPS", "")).zfill(5)) in ag_available:
            val_pairs_ag.append((m, e))
    val_pairs = val_pairs_ag
    print(f"  AG 过滤后: {len(val_pairs)} 样本")

    val_ag = AgricultureImageDataset([m for m, _ in val_pairs], train=False, seed=42)
    collate_fn = make_grid_collate_fn(global_stats, dynamic_feature_names)
    val_samples = build_grid_samples(val_pairs, soil_dict=soil_dict,
                                       dynamic_feature_names=dynamic_feature_names)
    val_dataset = GridTimeSeriesDataset(val_samples, ag_dataset=val_ag)
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                        collate_fn=collate_fn)

    # Load model
    ckpt = os.path.join(args.output_dir, "best_model.pth")
    model = TFTEncoderForYieldPrediction(
        soil_dim=SOIL_DIM, dynamic_feature_names=dynamic_feature_names,
        hidden_size=hidden_size, num_lstm_layers=num_lstm_layers,
        dropout=dropout, output_size=1, num_heads=num_heads,
        spatial_mode=spatial_mode,
        variable_selection_stage=hp.get("variable_selection_stage", "grid"),
        use_remote_sensing=use_rs,
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    # Run diagnosis on first batch
    for batch in loader:
        _capture_forward_remote_sensing(model, batch, device)
        break


if __name__ == "__main__":
    main()