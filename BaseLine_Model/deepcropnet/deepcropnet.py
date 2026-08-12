"""
DeepCropNet(DCN)基线(Lin et al 2020, ERL):AT-LSTM + MTL 区域输出层。

特征工程(对齐论文, B&H 2015):
  - 逐日县均值 -> GDD_d=max(0,Tmean-8°C), KDD_d=max(0,Tmax-30°C), PRCP=日降水
  - 从 4 月 1 日起连续 20 周,周累积 x_t=[GDD_t, KDD_t, PRCP_t]
  - 输入 z-score(训练集统计);目标 = 原始单产 bu/ac(可 --target anomaly 用趋势残差)

模型:3 层 LSTM(hidden=32)+ 单层 FC 时间注意力(H=Σ a_t·h_t)+ 按论文高温分区的
     3 个区域输出层(北 MN/WI/MI, 中 IA/IL/IN/OH, 南 MO/KY)。

数据/口径与 TFT 一致:DeepCropNet 9 玉米带州,训练 <2021,验证 2021。

用法:
  python deepcropnet.py                 # 原始单产目标
  python deepcropnet.py --target anomaly
  python deepcropnet.py --single_head   # 消融:无 MTL
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from common import data as D

OUT_DIR = _ROOT / "output"

# GDD/KDD 特征工程(B&H 2015;DCN 论文引用)
IDX_TMAX, IDX_TMIN, IDX_PRCP = 1, 2, 3
GDD_BASE, KDD_THRESH = 8.0, 30.0     # °C
START_DAY, N_WEEKS = 31, 20          # 4 月 1 日起 20 周
REGIONS = {"minnesota": 0, "wisconsin": 0, "michigan": 0,
           "iowa": 1, "illinois": 1, "indiana": 1, "ohio": 1,
           "missouri": 2, "kentucky": 2}


# ============================== 数据预处理 ==============================

def weekly_accumulate(x: np.ndarray, start_day: int, num_weeks: int) -> np.ndarray:
    out = np.zeros(num_weeks, dtype=np.float64)
    for w in range(num_weeks):
        a = start_day + 7 * w
        b = min(a + 7, len(x))
        out[w] = x[a:b].sum()
    return out


def county_gdd_kdd_prcp(feats):
    """feats: (G, T, 11) -> 县均值逐日 GDD/KDD/PRCP (T,)。"""
    tmax = feats[:, :, IDX_TMAX].mean(dim=0).numpy()
    tmin = feats[:, :, IDX_TMIN].mean(dim=0).numpy()
    prcp = feats[:, :, IDX_PRCP].mean(dim=0).numpy()
    tmean = (tmax + tmin) / 2.0
    gdd_d = np.clip(tmean - 273.15 - GDD_BASE, 0, None)
    kdd_d = np.clip(tmax - 273.15 - KDD_THRESH, 0, None)
    prcp_d = np.clip(prcp, 0, None)
    return gdd_d, kdd_d, prcp_d


def prepare_dcn(val_year=2021, out_dir=None, force=False):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_path, va_path = out_dir / "train_dcn_data.pt", out_dir / "val_dcn_data.pt"
    if tr_path.exists() and va_path.exists() and not force:
        tr = torch.load(tr_path, map_location="cpu", weights_only=False)
        va = torch.load(va_path, map_location="cpu", weights_only=False)
        print(f"[数据] 使用缓存 {tr_path}")
        return tr, va

    meta_lines = D.load_jsonl(D.DEFAULT_DATA_JSONL)
    cache = D.load_grid_cache(D.DEFAULT_GRID_CACHE)
    entries = cache["entries"]
    states = set(D.STATES)

    def build(samples):
        Xs, ys, regs, metas = [], [], [], []
        for m, e in samples:
            st = str(m.get("State", "")).lower()
            gdd_d, kdd_d, prcp_d = county_gdd_kdd_prcp(e["feats"])
            X = np.stack([
                weekly_accumulate(gdd_d, START_DAY, N_WEEKS),
                weekly_accumulate(kdd_d, START_DAY, N_WEEKS),
                weekly_accumulate(prcp_d, START_DAY, N_WEEKS),
            ], axis=1)                                   # (20, 3)
            Xs.append(X)
            ys.append(float(m["yield_per_acre"]))
            regs.append(REGIONS[st])
            metas.append({"fips": str(m["FIPS"]), "county": str(m.get("County", "")),
                          "state": st, "year": int(m["Year"])})
        return Xs, ys, regs, metas

    pairs = [(m, e) for m, e in zip(meta_lines, entries)
             if e is not None and str(m["State"]).lower() in states]
    tr_pairs = [p for p in pairs if int(p[0]["Year"]) < val_year]
    va_pairs = [p for p in pairs if int(p[0]["Year"]) == val_year]

    Xtr, ytr, reg_tr, meta_tr = build(tr_pairs)
    Xva, yva, reg_va, meta_va = build(va_pairs)
    Xtr = np.stack(Xtr).astype(np.float32)
    Xva = np.stack(Xva).astype(np.float32)
    ytr = np.array(ytr, dtype=np.float32)
    yva = np.array(yva, dtype=np.float32)
    print(f"[数据] 训练 {len(Xtr)} / 验证 {len(Xva)} (20周×3特征 GDD/KDD/PRCP)")

    mean = Xtr.reshape(-1, 3).mean(0)
    std = Xtr.reshape(-1, 3).std(0) + 1e-8
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    # 池化线性趋势(供 anomaly 目标;仅 4 训练年,趋势本身噪声大,默认用 raw)
    years_tr = np.array([m["year"] for m in meta_tr], dtype=np.float64)
    A = np.stack([years_tr, np.ones_like(years_tr)], axis=1)
    coef, *_ = np.linalg.lstsq(A, ytr.astype(np.float64), rcond=None)
    ytr_trend = (coef[0] * years_tr + coef[1]).astype(np.float32)
    yva_trend = (coef[0] * np.array([m["year"] for m in meta_va]) + coef[1]).astype(np.float32)

    def save(path, X, y, trend, reg, metas):
        torch.save({"X": X, "y_raw": y, "y_trend": trend,
                    "region": np.array(reg, dtype=np.int64), "meta": metas}, path)
    save(tr_path, Xtr, ytr, ytr_trend, reg_tr, meta_tr)
    save(va_path, Xva, yva, yva_trend, reg_va, meta_va)
    print(f"[数据] 已保存 {tr_path} / {va_path}")
    print("区域分布 train:", {r: reg_tr.count(r) for r in sorted(set(reg_tr))})
    print("区域分布 val:  ", {r: reg_va.count(r) for r in sorted(set(reg_va))})
    return torch.load(tr_path, map_location="cpu", weights_only=False), \
        torch.load(va_path, map_location="cpu", weights_only=False)


# ============================== 模型 ==============================

class ATLSTM(nn.Module):
    """3 层 LSTM + 单层 FC 时间注意力(论文式6/7)。"""
    def __init__(self, n_features=3, hidden=32, n_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers, batch_first=True)
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)              # (B, T, hidden)
        a = F.softmax(self.attn(h), dim=1)   # 时间维归一
        H = (h * a).sum(dim=1)           # (B, hidden)
        return H, a


class DeepCropNet(nn.Module):
    """AT-LSTM + 区域专用 MTL 输出层。region 按样本给定(B,)。"""
    def __init__(self, region_ids, n_features=3, hidden=32, n_layers=3):
        super().__init__()
        self.region_ids = sorted(int(r) for r in region_ids)
        self.temporal = ATLSTM(n_features, hidden, n_layers)
        self.heads = nn.ModuleDict({str(r): nn.Linear(hidden, 1) for r in self.region_ids})

    def forward(self, x, region):
        H, attn_w = self.temporal(x)     # (B, hidden), (B, T, 1)
        B = H.size(0)
        out = torch.zeros(B, 1, device=H.device, dtype=H.dtype)
        for r in self.region_ids:
            mask = region == r
            if mask.any():
                out[mask] = self.heads[str(r)](H[mask])
        return out, attn_w


# ============================== 训练 ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_year", type=int, default=2021)
    ap.add_argument("--target", type=str, default="raw", choices=["raw", "anomaly"])
    ap.add_argument("--single_head", action="store_true", help="消融:无 MTL")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--force_prep", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    tr, va = prepare_dcn(args.val_year, out_dir, args.force_prep)

    # 目标:raw = 原始单产;anomaly = 原始单产 - 趋势。预测后还原。
    if args.target == "raw":
        bias_tr = np.zeros_like(tr["y_raw"], dtype=np.float32)
        bias_va = np.zeros_like(va["y_raw"], dtype=np.float32)
    else:
        bias_tr = np.asarray(tr["y_trend"], dtype=np.float32)
        bias_va = np.asarray(va["y_trend"], dtype=np.float32)
    ytr_target = np.asarray(tr["y_raw"], dtype=np.float32) - bias_tr
    yva_target = np.asarray(va["y_raw"], dtype=np.float32) - bias_va

    y_mean = float(ytr_target.mean())
    y_std = float(ytr_target.std()) + 1e-6
    ytr_t = (ytr_target - y_mean) / y_std
    yva_t = (yva_target - y_mean) / y_std

    region_tr = torch.tensor(tr["region"], dtype=torch.long)
    region_va = torch.tensor(va["region"], dtype=torch.long)
    if args.single_head:
        region_tr = torch.zeros_like(region_tr)
        region_va = torch.zeros_like(region_va)

    region_ids = sorted(set(tr["region"].tolist()) | set(va["region"].tolist()))
    model = DeepCropNet(region_ids=region_ids).to(device)
    print(f"[模型] DeepCropNet 参数: {sum(p.numel() for p in model.parameters()):,}")

    Xtr = torch.tensor(tr["X"], dtype=torch.float32, device=device)
    Xva = torch.tensor(va["X"], dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr_t, dtype=torch.float32, device=device).unsqueeze(1)
    N = len(Xtr)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best_rmse, best_state, patience = float("inf"), None, args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        nb = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            out, _ = model(Xtr[idx], region_tr[idx])
            loss = loss_fn(out, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1

        model.eval()
        with torch.no_grad():
            pv, _ = model(Xva, region_va)
            pred_raw = pv.squeeze(1).cpu().numpy() * y_std + y_mean + bias_va
        m = D.metrics(pred_raw, np.asarray(va["y_raw"]))
        if m["rmse"] < best_rmse - 1e-6:
            best_rmse = m["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d} loss={tot/nb:.3f} valRMSE={m['rmse']:.3f}")
        if patience <= 0:
            print(f"    early stop @ {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv, _ = model(Xva, region_va)
        pred_raw = pv.squeeze(1).cpu().numpy() * y_std + y_mean + bias_va
    overall = D.metrics(pred_raw, np.asarray(va["y_raw"]))
    print(f"  ==== DeepCropNet({args.target}): RMSE={overall['rmse']:.3f} "
          f"R²={overall['r2']:.3f} Corr={overall['corr']:.3f}")

    per_region = {}
    for r in sorted(set(va["region"].tolist())):
        mask = va["region"] == r
        sts = ",".join(sorted(set(m["state"] for m in np.array(va["meta"])[mask])))
        mr = D.metrics(pred_raw[mask], np.asarray(va["y_raw"])[mask])
        mr["states"] = sts
        per_region[r] = mr
        print(f"    region {r} [{sts}]: n={mr['n']} RMSE={mr['rmse']:.3f} R²={mr['r2']:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / f"best_dcn_{args.target}.pth")
    result = {"model": "deepcropnet", "target": args.target, "overall": overall,
              "per_region": {str(k): v for k, v in per_region.items()}}
    with open(out_dir / f"dcn_results_{args.target}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {out_dir / ('best_dcn_' + args.target + '.pth')}")


if __name__ == "__main__":
    main()
