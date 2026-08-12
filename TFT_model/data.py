"""
载入 cropnet 数据集（dataset.jsonl），构造 DataLoader。

参考 Product_model/all_product/data.py，根据 cropnet 数据特点适配：
  - 动态特征：WRF-HRRR 气象 11 维（无 MODIS 遥感）
  - 静态特征：carbon_bucket + ph_bucket（来自土壤映射表）
  - 无 crop_phase、无 pred、无 prefix_month

用法：
  from data import TimeSeriesDataset, make_custom_collate_fn, ...
  dataset = TimeSeriesDataset(jsonl_path)
  loader = DataLoader(dataset, batch_size=32, collate_fn=make_custom_collate_fn(stats))
"""

import json
import math
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from typing import Any, List, Dict, Optional, Tuple, Iterator

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "train_dataset")

DEFAULT_DATA_JSONL = os.path.join(TRAIN_DATA_DIR, "dataset.jsonl")
DEFAULT_GRID_CACHE = os.path.join(TRAIN_DATA_DIR, "grid_cache.pt")
# 源数据已迁移到 DataSrc/(2026-08 重构)
DEFAULT_COUNTY_SOIL = os.path.join(SCRIPT_DIR, "..", "DataSrc", "soil_dataset", "county_soil.json")

# 县级土壤特征(gSSURGO,连续值,0-30cm 加权)
SOIL_FEATURES = ["clay_pct", "sand_pct", "silt_pct", "om_pct", "ph", "bulk_density", "awc"]
SOIL_DIM = len(SOIL_FEATURES)

DEFAULT_CARBON_BUCKET_ID_PATH = os.path.join(TRAIN_DATA_DIR, "us_carbon_bucket_id.json")
DEFAULT_PH_BUCKET_ID_PATH = os.path.join(TRAIN_DATA_DIR, "us_ph_bucket_id.json")

# ============================================================
# 动态特征列（WRF-HRRR 原始列名）
# ============================================================
DEFAULT_DYNAMIC_FEATURE_NAMES = [
    "Avg Temperature (K)",
    "Max Temperature (K)",
    "Min Temperature (K)",
    "Precipitation (kg m**-2)",
    "Relative Humidity (%)",
    "Wind Gust (m s**-1)",
    "Wind Speed (m s**-1)",
    "U Component of Wind (m s**-1)",
    "V Component of Wind (m s**-1)",
    "Downward Shortwave Radiation Flux (W m**-2)",
    "Vapor Pressure Deficit (kPa)",
]

# 累计积温通道（可选，--use_gdd 时追加到动态特征末尾）
GDD_FEATURE_NAME = "CumGDD"
GDD_BASE = 8.0  # °C，与 DeepCropNet 基线一致

# 农学构造特征（--use_constructed 时全部追加；全部由原始 11 通道 + 网格坐标/日期现算）
KDD_FEATURE_NAME = "KDD"
CUMPRCP_FEATURE_NAME = "CumPRCP"
CUMDEFICIT_FEATURE_NAME = "CumDeficit"
KDD_BASE = 30.0  # °C，极端高温阈值（DeepCropNet / Butler & Huybers 2015）
# 构造特征统一顺序：--use_constructed 时按此顺序追加到 11 维原始气象之后（11+4=15 维）
CONSTRUCTED_FEATURES = [
    GDD_FEATURE_NAME,
    KDD_FEATURE_NAME,
    CUMPRCP_FEATURE_NAME,
    CUMDEFICIT_FEATURE_NAME,
]
_GSC = 0.0820  # 太阳常数 (MJ m^-2 min^-1)，FAO-56
_LAMBDA = 2.45  # 蒸发潜热 (MJ kg^-1)，用于 Ra MJ→mm 水当量


def append_cum_gdd(feats: torch.Tensor) -> torch.Tensor:
    """在 feats (G,T,F) 末尾追加累计积温通道 CumGDD。

    由每网格逐日均温(通道 0,单位 K)计算:
      Tc[g,t]  = Tmean[g,t] - 273.15
      gdd[g,t] = max(0, Tc[g,t] - GDD_BASE)
      GDD[g,t] = Σ_{s=1..t} gdd[g,s]      # 按网格累计
    返回 (G,T,F+1)，GDD 为最后一通道。
    """
    tmean_c = feats[..., 0] - 273.15            # (G,T) 摄氏
    gdd_daily = torch.clamp(tmean_c - GDD_BASE, min=0.0)
    cum = gdd_daily.cumsum(dim=-1)              # (G,T) 累计积温
    return torch.cat([feats, cum.unsqueeze(-1)], dim=-1)


def hargreaves_pet(
    tmean_c: torch.Tensor,
    tmax_c: torch.Tensor,
    tmin_c: torch.Tensor,
    lat_deg: torch.Tensor,
    doy_start: int = 60,
) -> torch.Tensor:
    """Hargreaves 参考蒸散 ET0（mm/day），逐网格逐日。

    参数均为 (G, T)：tmean/tmax/tmin 摄氏温度；lat_deg: (G,) 纬度（度）。
    序列自 3/1 起 275 天，故首日 doy=60（非闰年），doy_start=60 使 3/1 落在第 60 天；
    忽略闰年差异（2020 为闰年，逐日太阳几何误差 <1%）。
    返回 (G, T) mm/day。
    """
    T = tmean_c.shape[-1]
    doy = torch.arange(T, dtype=tmean_c.dtype, device=tmean_c.device) + doy_start  # (T,)
    lat = torch.deg2rad(lat_deg).unsqueeze(-1)                       # (G,1)
    dr = 1.0 + 0.033 * torch.cos(2.0 * math.pi * doy / 365.0)        # (T,)
    delta = 0.409 * torch.sin(2.0 * math.pi * doy / 365.0 - 1.39)    # (T,) 太阳赤纬 rad
    sin_phi, cos_phi = torch.sin(lat), torch.cos(lat)                # (G,1)
    sin_delta, cos_delta = torch.sin(delta), torch.cos(delta)        # (T,)
    # 日落时角 ωs = arccos(-tan φ · tan δ)，钳制到 [-1,1] 防数值越界
    omega = torch.acos(torch.clamp(-torch.tan(lat) * torch.tan(delta), -1.0, 1.0))  # (G,T)
    ra = (24.0 * 60.0 / math.pi) * _GSC * dr * (
        omega * sin_phi * sin_delta + cos_phi * cos_delta * torch.sin(omega)
    )                                                                # (G,T) MJ/m^2/day
    # Hargreaves-Samani: 0.0023 常数的 Ra 须为 mm/day 水当量, 故 MJ/m^2/day 除以 λ
    et0 = 0.0023 * (ra / _LAMBDA) * (tmean_c + 17.8) * torch.sqrt(torch.clamp(tmax_c - tmin_c, min=0.0))
    return et0


def append_constructed_features(
    feats: torch.Tensor,
    coords: Optional[torch.Tensor] = None,
    names: Optional[List[str]] = None,
) -> torch.Tensor:
    """在 feats（G,T,F0）末尾按 names 顺序追加构造通道，返回（G,T,F0+K）。

    构造通道（逐网格逐日）：
      CumGDD     = Σ max(0, Tmean−8)           有效热量累积
      KDD        = Σ max(0, Tmean−30)          极端高温（热害）累积
      CumPRCP    = Σ max(0, precip)            水分供给累积
      CumDeficit = CumPRCP − Σ ET0(Hargreaves) 累计水分亏缺（负值=干旱），需 coords 提供纬度
    names 缺省为全部 CONSTRUCTED_FEATURES；只含 CumGDD 时与 append_cum_gdd 等价。
    """
    names = list(names or CONSTRUCTED_FEATURES)
    if not names:
        return feats
    tmean_c = feats[..., 0] - 273.15            # (G,T)
    tmax_c = feats[..., 1] - 273.15
    tmin_c = feats[..., 2] - 273.15
    prcp = torch.clamp(feats[..., 3], min=0.0)  # (G,T) mm/day,降水非负 (index 3 = Precipitation,非 RH)
    out = [feats]
    for n in names:
        if n == GDD_FEATURE_NAME:
            ch = torch.cumsum(torch.clamp(tmean_c - GDD_BASE, min=0.0), dim=-1)
        elif n == KDD_FEATURE_NAME:
            ch = torch.cumsum(torch.clamp(tmean_c - KDD_BASE, min=0.0), dim=-1)
        elif n == CUMPRCP_FEATURE_NAME:
            ch = torch.cumsum(prcp, dim=-1)
        elif n == CUMDEFICIT_FEATURE_NAME:
            if coords is None:
                raise ValueError("CumDeficit 需要网格坐标(coords)计算 Hargreaves 蒸散")
            pet = hargreaves_pet(tmean_c, tmax_c, tmin_c, coords[..., 0])
            ch = torch.cumsum(prcp, dim=-1) - torch.cumsum(pet, dim=-1)
        else:
            raise ValueError(f"未知构造特征: {n}")
        out.append(ch.unsqueeze(-1))            # 保持与 feats 同秩, 便于 cat
    return torch.cat(out, dim=-1)


def dynamic_names_from_hparams(hp, base=None):
    """按训练 hparams 决定动态特征名列表(与 train.py 的开关逻辑一致)。

    hp: model_hparams.json 的内容(dict,可能缺键)。
    use_constructed=True -> 11 原始 + 全部 CONSTRUCTED_FEATURES(15 维);
    否则 use_gdd=True -> 11 原始 + CumGDD(12 维,旧口径);否则仅 11 原始。
    """
    names = list(base if base is not None else DEFAULT_DYNAMIC_FEATURE_NAMES)
    hp = hp or {}
    if hp.get("use_constructed", False):
        names += [n for n in CONSTRUCTED_FEATURES if n not in names]
    elif hp.get("use_gdd", False):
        if GDD_FEATURE_NAME not in names:
            names.append(GDD_FEATURE_NAME)
    return names


# ============================================================
# Bucket ID 映射加载
# ============================================================
def load_bucket_id_map(path: str) -> Dict[str, int]:
    """读取 JSON {字符串: ID} 映射，key 去空格。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k).strip(): int(v) for k, v in raw.items()}


def bucket_vocab_size(bucket_map: Dict[str, int]) -> int:
    """Embedding 行数 = max_id + 1 (padding/UNK)。"""
    if not bucket_map:
        return 1
    mx = max(int(v) for v in bucket_map.values())
    return mx + 2


# ============================================================
# Dataset
# ============================================================
class TimeSeriesDataset(Dataset):
    """
    cropnet 时序数据集。

    每个 jsonl 样本输出一条全长序列（不做滑窗切割）：
      - dynamic: (T, F) 气象时间序列
      - static_bucket_ids: {"carbon_bucket": (1,), "ph_bucket": (1,)}
      - yield_per_acre: (1,) 单产 (bu/ac)
      - seq_len: int 有效时间步数
    """

    def __init__(
        self,
        samples: List[Dict],
        dynamic_feature_names: Optional[List[str]] = None,
        carbon_bucket_id_path: str = DEFAULT_CARBON_BUCKET_ID_PATH,
        ph_bucket_id_path: str = DEFAULT_PH_BUCKET_ID_PATH,
        seed: int = 42,
    ):
        if dynamic_feature_names is None:
            dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.dynamic_feature_names = list(dynamic_feature_names)

        # 加载 bucket → ID 映射
        self.carbon_to_id = load_bucket_id_map(carbon_bucket_id_path)
        self.ph_to_id = load_bucket_id_map(ph_bucket_id_path)
        self._carbon_pad_id = max(self.carbon_to_id.values()) + 1 if self.carbon_to_id else 0
        self._ph_pad_id = max(self.ph_to_id.values()) + 1 if self.ph_to_id else 0

        # 预处理：jsonl 行 → 子样本
        self.sub_samples = []
        all_seq_lens = []

        for raw in samples:
            state = raw.get("State", "unknown")
            seq_len = int(raw.get("l_enc", 0))
            if seq_len == 0:
                continue

            # === 动态特征 ===
            cols = []
            for feat in self.dynamic_feature_names:
                feat_data = raw.get(feat)
                if feat_data is None or not isinstance(feat_data, list):
                    raise ValueError(
                        f"样本 State={state} 缺少或格式错误的动态特征: {feat}"
                    )
                if len(feat_data) != seq_len:
                    # 长度不一致时截断到最短
                    seq_len = min(seq_len, len(feat_data))
                cols.append(torch.tensor(feat_data[:seq_len], dtype=torch.float32))
            dynamic_data = torch.stack(cols, dim=1)  # (T, F)

            # === 月份（逐时间步，用于 embedding）===
            month_raw = raw.get("month")
            if month_raw is None or not isinstance(month_raw, list):
                raise ValueError(f"样本 State={state} 缺少 month 字段")
            month_t = torch.tensor(month_raw[:seq_len], dtype=torch.long)

            # === 静态特征 ===
            carbon_key = str(raw.get("carbon_bucket", "")).strip()
            ph_key = str(raw.get("ph_bucket", "")).strip()
            carbon_id = self.carbon_to_id.get(carbon_key, self._carbon_pad_id)
            ph_id = self.ph_to_id.get(ph_key, self._ph_pad_id)
            static_bucket_ids = {
                "carbon_bucket": torch.tensor([carbon_id], dtype=torch.long),
                "ph_bucket": torch.tensor([ph_id], dtype=torch.long),
            }

            # === 目标:单产 yield_per_acre (bu/ac) ===
            ypa_raw = raw.get("yield_per_acre")
            if ypa_raw is None:
                raise ValueError(f"样本 State={state} 缺少 yield_per_acre 字段")
            yield_t = torch.tensor([float(ypa_raw)], dtype=torch.float32)

            self.sub_samples.append({
                "dynamic": dynamic_data,
                "month": month_t,
                "static_bucket_ids": static_bucket_ids,
                "yield_per_acre": yield_t,
                "seq_len": seq_len,
                "state": state,
                "year": int(raw.get("Year", 0)),
                "FIPS": str(raw.get("FIPS", "")),
                "County": str(raw.get("County", "")),
            })
            all_seq_lens.append(seq_len)

        self.max_seq_len = max(all_seq_lens) if all_seq_lens else 0

    def __len__(self):
        return len(self.sub_samples)

    def __getitem__(self, idx):
        s = self.sub_samples[idx]
        return (
            s["dynamic"],          # (T, F)
            s["month"],            # (T,) long
            s["static_bucket_ids"], # {"carbon_bucket": (1,), "ph_bucket": (1,)}
            s["yield_per_acre"],   # (1,) 单产 bu/ac
            s["seq_len"],
            s["state"],
            s["year"],
            s["FIPS"],
            s["County"],
        )


# ============================================================
# 标准化工具
# ============================================================
def _apply_standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对任意 N 维张量用 mean(最后一维)、std(最后一维) 标准化。

    - (T, F)   -> mean/std view (1, F)   广播
    - (G, T, F)-> mean/std view (1, 1, F) 广播
    """
    if x.numel() == 0:
        return x
    shape = [1] * (x.ndim - 1) + [-1]
    mean = mean.to(device=x.device, dtype=x.dtype).view(*shape)
    std = std.to(device=x.device, dtype=x.dtype).view(*shape).clamp_min(eps)
    return (x - mean) / std


def compute_global_dynamic_stats(
    train_samples: List[Dict],
    dynamic_feature_names: Optional[List[str]] = None,
    eps: float = 1e-6,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    在训练集上逐时间步池化，计算每维动态特征的全局 mean/std。
    返回 {"dynamic": (mean (F,), std (F,))}。
    """
    if dynamic_feature_names is None:
        dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)
    F = len(dynamic_feature_names)
    sum_x = torch.zeros(F)
    sumsq_x = torch.zeros(F)
    n_steps = 0

    for raw in train_samples:
        cols = []
        for feat in dynamic_feature_names:
            arr = raw.get(feat)
            if not isinstance(arr, list):
                continue
            cols.append(torch.tensor(arr, dtype=torch.float32))
        if not cols:
            continue
        x = torch.stack(cols, dim=1)  # (T, F)
        n_steps += x.shape[0]
        sum_x += x.sum(dim=0)
        sumsq_x += (x * x).sum(dim=0)

    if n_steps == 0:
        raise ValueError("无有效时间步，无法计算全局统计量")

    mean = sum_x / n_steps
    var = sumsq_x / n_steps - mean * mean
    std = torch.sqrt(torch.clamp(var, min=0.0)).clamp_min(eps)
    return {"dynamic": (mean, std)}


def compute_state_dynamic_stats(
    train_samples: List[Dict],
    dynamic_feature_names: Optional[List[str]] = None,
    eps: float = 1e-6,
) -> Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    按州分别计算动态特征 mean/std。
    返回 {state: {"dynamic": (mean (F,), std (F,))}}。
    """
    if dynamic_feature_names is None:
        dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)
    F = len(dynamic_feature_names)
    sum_by_state: Dict[str, torch.Tensor] = {}
    sumsq_by_state: Dict[str, torch.Tensor] = {}
    n_by_state: Dict[str, int] = {}

    for raw in train_samples:
        st = str(raw.get("State", ""))
        cols = []
        for feat in dynamic_feature_names:
            arr = raw.get(feat)
            if not isinstance(arr, list):
                break
            cols.append(torch.tensor(arr, dtype=torch.float32))
        if len(cols) != F:
            continue
        x = torch.stack(cols, dim=1)

        if st not in n_by_state:
            sum_by_state[st] = torch.zeros(F)
            sumsq_by_state[st] = torch.zeros(F)
            n_by_state[st] = 0
        sum_by_state[st] += x.sum(dim=0)
        sumsq_by_state[st] += (x * x).sum(dim=0)
        n_by_state[st] += x.shape[0]

    out = {}
    for st, n in n_by_state.items():
        mean = sum_by_state[st] / n
        var = sumsq_by_state[st] / n - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0)).clamp_min(eps)
        out[st] = {"dynamic": (mean, std)}
    return out


def save_feature_norm_json(
    path: str,
    stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    per_state: bool = False,
    state_stats: Optional[Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]] = None,
) -> None:
    """保存标准化统计量到 JSON。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if per_state and state_stats:
        payload: Dict = {"mode": "per_state", "states": {}}
        for st, s in state_stats.items():
            payload["states"][st] = {}
            for k, (m, s_) in s.items():
                payload["states"][st][k] = {
                    "mean": m.detach().cpu().tolist(),
                    "std": s_.detach().cpu().tolist(),
                }
        if stats:
            payload["fallback"] = {}
            for k, (m, s_) in stats.items():
                payload["fallback"][k] = {
                    "mean": m.detach().cpu().tolist(),
                    "std": s_.detach().cpu().tolist(),
                }
    else:
        payload = {}
        for k, (m, s_) in stats.items():
            payload[k] = {"mean": m.detach().cpu().tolist(), "std": s_.detach().cpu().tolist()}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_feature_norm_json(
    path: str,
) -> Tuple[Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]], Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]:
    """
    读取标准化统计量 JSON。
    返回 (state_stats, fallback_stats)。
    - 若为 per_state 格式：state_stats 按州索引，fallback 为全局备选
    - 若为全局格式：state_stats 为空字典，fallback 为全局统计
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    state_stats: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    fallback: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None

    if "states" in raw:
        for st, s in raw["states"].items():
            state_stats[st] = {}
            for k, block in s.items():
                state_stats[st][k] = (
                    torch.tensor(block["mean"], dtype=torch.float32),
                    torch.tensor(block["std"], dtype=torch.float32),
                )
        if "fallback" in raw:
            fallback = {}
            for k, block in raw["fallback"].items():
                fallback[k] = (
                    torch.tensor(block["mean"], dtype=torch.float32),
                    torch.tensor(block["std"], dtype=torch.float32),
                )
    else:
        fallback = {}
        for k, block in raw.items():
            fallback[k] = (
                torch.tensor(block["mean"], dtype=torch.float32),
                torch.tensor(block["std"], dtype=torch.float32),
            )

    return state_stats, fallback


# ============================================================
# Collate 函数
# ============================================================
def make_custom_collate_fn(
    global_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    dynamic_feature_names: Optional[List[str]] = None,
    eps: float = 1e-6,
):
    """
    全局标准化 + 补零 collate。
    global_stats 需包含键 "dynamic" -> (mean (F,), std (F,))。
    """
    mean_d, std_d = global_stats["dynamic"]
    feat_names = list(dynamic_feature_names or DEFAULT_DYNAMIC_FEATURE_NAMES)

    def collate_fn(batch: List[Tuple]) -> Tuple:
        dynamic_list = [item[0] for item in batch]
        month_list = [item[1] for item in batch]   # (T,) long
        static_list = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        seq_lens = torch.tensor([item[4] for item in batch], dtype=torch.long)
        states = [item[5] for item in batch]
        years = [int(item[6]) for item in batch]
        fips_list = [item[7] for item in batch]
        county_list = [item[8] for item in batch]

        # 标准化
        dynamic_list = [_apply_standardize(x, mean_d, std_d, eps) for x in dynamic_list]

        # 补零到 batch 最大长度
        B = len(dynamic_list)
        F = len(feat_names)
        max_len = int(seq_lens.max().item())

        # 动态特征补零
        padded = torch.zeros(B, max_len, F, dtype=torch.float32)
        for i, x in enumerate(dynamic_list):
            sl = x.shape[0]
            padded[i, :sl] = x

        # month 补零（用 0 作为 padding month）
        month_padded = torch.zeros(B, max_len, dtype=torch.long)
        for i, m in enumerate(month_list):
            sl = m.shape[0]
            month_padded[i, :sl] = m

        # 按列拆分
        grouped_feats = {name: padded[:, :, j:j+1] for j, name in enumerate(feat_names)}

        # 静态特征
        static_bucket_ids = {
            "carbon_bucket": torch.stack([s["carbon_bucket"] for s in static_list], dim=0),
            "ph_bucket": torch.stack([s["ph_bucket"] for s in static_list], dim=0),
        }
        labels = torch.stack(labels, dim=0)

        return (
            grouped_feats,
            month_padded,      # (B, T) long
            static_bucket_ids,
            labels,
            seq_lens,
            states,
            years,
            fips_list,
            county_list,
        )

    return collate_fn


def make_state_custom_collate_fn(
    state_feature_stats: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    fallback_feature_stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    dynamic_feature_names: Optional[List[str]] = None,
    eps: float = 1e-6,
):
    """
    按州标准化 + 补零 collate。
    每个样本按其 state 选取对应的 mean/std。
    """
    feat_names = list(dynamic_feature_names or DEFAULT_DYNAMIC_FEATURE_NAMES)

    def collate_fn(batch: List[Tuple]) -> Tuple:
        dynamic_list = [item[0] for item in batch]
        month_list = [item[1] for item in batch]
        static_list = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        seq_lens = torch.tensor([item[4] for item in batch], dtype=torch.long)
        states = [item[5] for item in batch]
        years = [int(item[6]) for item in batch]
        fips_list = [item[7] for item in batch]
        county_list = [item[8] for item in batch]

        for i, st in enumerate(states):
            stats_i = state_feature_stats.get(st)
            if stats_i is None:
                if fallback_feature_stats is None:
                    raise KeyError(f"state_feature_stats 缺失州 {st}，且无 fallback")
                stats_i = fallback_feature_stats
            mean_d, std_d = stats_i["dynamic"]
            dynamic_list[i] = _apply_standardize(dynamic_list[i], mean_d, std_d, eps)

        B = len(dynamic_list)
        F = len(feat_names)
        max_len = int(seq_lens.max().item())
        padded = torch.zeros(B, max_len, F, dtype=torch.float32)
        for i, x in enumerate(dynamic_list):
            sl = x.shape[0]
            padded[i, :sl] = x

        # month 补零
        month_padded = torch.zeros(B, max_len, dtype=torch.long)
        for i, m in enumerate(month_list):
            sl = m.shape[0]
            month_padded[i, :sl] = m

        grouped_feats = {name: padded[:, :, j:j+1] for j, name in enumerate(feat_names)}
        static_bucket_ids = {
            "carbon_bucket": torch.stack([s["carbon_bucket"] for s in static_list], dim=0),
            "ph_bucket": torch.stack([s["ph_bucket"] for s in static_list], dim=0),
        }
        labels = torch.stack(labels, dim=0)

        return (
            grouped_feats,
            month_padded,
            static_bucket_ids,
            labels,
            seq_lens,
            states,
            years,
            fips_list,
            county_list,
        )

    return collate_fn


# ============================================================
# Samplers
# ============================================================
class PerRawSampleBatchSampler(Sampler):
    """
    按原始样本（State, Year）分组，确保同一组的所有子样本进入同一个 batch（或顺序相邻的 batch）。

    对 cropnet 数据集（每个 jsonl 行就是一个子样本，无滑窗倍增），
    每个 (State, Year) 组只包含一条样本，因此等价于普通随机采样。
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.raw_sample_groups = {}
        for sub_idx, sub_sample in enumerate(dataset.sub_samples):
            raw_id = (sub_sample["state"], sub_sample["year"])
            self.raw_sample_groups.setdefault(raw_id, []).append(sub_idx)

        self.all_raw_sample_ids = list(self.raw_sample_groups.keys())
        if self.shuffle:
            random.shuffle(self.all_raw_sample_ids)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class PerYearDistinctStateBatchSampler(Sampler):
    """
    每个 batch 内：
      1) 样本来自同一年
      2) 每个州最多出现一次
    适用于 cropnet 数据集做 yearly state-balance 采样。
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.year_state_groups: Dict[int, Dict[str, List[int]]] = {}
        for idx, s in enumerate(dataset.sub_samples):
            y = int(s["year"])
            st = str(s["state"])
            self.year_state_groups.setdefault(y, {}).setdefault(st, []).append(idx)

        self.years = sorted(self.year_state_groups.keys())
        if not self.years:
            raise ValueError("PerYearDistinctStateBatchSampler: 数据集为空")

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        years = list(self.years)
        if self.shuffle:
            rng.shuffle(years)

        for year in years:
            st_pools: Dict[str, List[int]] = {
                st: list(idxs) for st, idxs in self.year_state_groups[year].items()
            }
            if self.shuffle:
                for pool in st_pools.values():
                    rng.shuffle(pool)

            active = [st for st, pool in st_pools.items() if pool]
            if self.shuffle:
                rng.shuffle(active)

            while active:
                batch = []
                next_active = []
                for st in active:
                    pool = st_pools[st]
                    if not pool:
                        continue
                    batch.append(pool.pop())
                    if pool:
                        next_active.append(st)
                    if len(batch) == self.batch_size:
                        break
                if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                    yield batch
                active = next_active

    def __len__(self) -> int:
        total = 0
        for _, st_groups in self.year_state_groups.items():
            max_windows = max(len(v) for v in st_groups.values())
            total += max_windows
        return total


# ============================================================
# 加载辅助
# ============================================================
def load_jsonl(path: str) -> List[Dict]:
    """读取 JSONL 文件，返回 list[dict]。"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def create_dataloader(
    jsonl_path: str = DEFAULT_DATA_JSONL,
    batch_size: int = 32,
    shuffle: bool = True,
    dynamic_feature_names: Optional[List[str]] = None,
    sampler_type: str = "default",
    **kwargs,
) -> Tuple[DataLoader, TimeSeriesDataset, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    一键创建 DataLoader（自动计算全局标准化统计量）。

    参数：
      sampler_type: "default" | "per_year_state"
    """
    samples = load_jsonl(jsonl_path)
    if dynamic_feature_names is None:
        dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)

    dataset = TimeSeriesDataset(samples, dynamic_feature_names=dynamic_feature_names)

    # 在训练集上计算全局标准化统计量
    global_stats = compute_global_dynamic_stats(samples, dynamic_feature_names)

    collate_fn = make_custom_collate_fn(global_stats, dynamic_feature_names)

    if sampler_type == "per_year_state":
        sampler = PerYearDistinctStateBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    else:
        sampler = PerRawSampleBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    return loader, dataset, global_stats


# ============================================================
# 网格级（9×9 km）数据 —— 对齐 MMST-ViT 的空间聚合
# ============================================================
def load_grid_cache(path: str = DEFAULT_GRID_CACHE) -> Dict:
    """加载 grid_cache.pt,返回 {"version", "feat_names", "entries"}。"""
    payload = torch.load(path, map_location="cpu")
    if payload.get("version") != 3 or payload.get("coord_type") != "grid_center":
        raise ValueError("grid_cache must use version 3 with coord_type='grid_center'")
    return payload


def compute_grid_global_stats(
    pairs: List[Tuple[Dict, Dict]],
    dynamic_feature_names: Optional[List[str]] = None,
    eps: float = 1e-6,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    在网格级样本对 (meta, entry) 上计算 11 维气象的全局 mean/std。
    pairs 需已按训练年份过滤(避免验证集泄漏)。
    返回 {"dynamic": (mean (F,), std (F,))} —— 与 compute_global_dynamic_stats 同格式。
    """
    if dynamic_feature_names is None:
        dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)
    F = len(dynamic_feature_names)
    constructed = [n for n in CONSTRUCTED_FEATURES if n in dynamic_feature_names]
    sum_x = torch.zeros(F)
    sumsq_x = torch.zeros(F)
    n_steps = 0
    for _, entry in pairs:
        if entry is None:
            continue
        x = entry["feats"]  # (G, T, F)
        if constructed and x.shape[-1] < F:
            x = append_constructed_features(x, coords=entry.get("coords"), names=constructed)
        sum_x += x.sum(dim=(0, 1))
        sumsq_x += (x * x).sum(dim=(0, 1))
        n_steps += int(x.shape[0]) * int(x.shape[1])

    if n_steps == 0:
        raise ValueError("无有效网格时间步，无法计算全局统计量")

    mean = sum_x / n_steps
    var = sumsq_x / n_steps - mean * mean
    std = torch.sqrt(torch.clamp(var, min=0.0)).clamp_min(eps)
    return {"dynamic": (mean, std)}


def load_county_soil(path: str = DEFAULT_COUNTY_SOIL) -> Dict[str, Dict[str, float]]:
    """加载 county_soil.json -> {FIPS: {clay_pct: .., sand_pct: .., ...}}。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_soil_stats(
    soil_dict: Dict[str, Dict[str, float]],
    fips_subset: Optional[set] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """在指定县子集上计算土壤特征的全局 mean/std(用于 z-score 标准化)。

    fips_subset: 若给定(如训练县 FIPS 集合),只在该子集上算统计量,
    与动态特征只在训练年份上算保持一致(避免验证县泄漏)。
    默认 None = 全部县。
    """
    keys = soil_dict.keys() if fips_subset is None else [
        f for f in soil_dict if f in fips_subset
    ]
    arr = np.array(
        [[soil_dict[f][var] for var in SOIL_FEATURES] for f in keys
         if all(soil_dict[f].get(var) is not None for var in SOIL_FEATURES)],
        dtype=np.float32,
    )
    if arr.shape[0] == 0:
        raise ValueError("county_soil.json 无有效土壤记录")
    mean = torch.from_numpy(arr.mean(axis=0))
    std = torch.from_numpy(arr.std(axis=0)).clamp_min(eps)
    return mean, std


def build_grid_samples(
    pairs: List[Tuple[Dict, Dict]],
    soil_dict: Dict[str, Dict[str, float]],
    dynamic_feature_names: Optional[List[str]] = None,
) -> List[Dict]:
    """
    把 (meta, entry) 对转成 GridTimeSeriesDataset 的子样本列表。

    静态土壤特征(县级,连续 7 维,不分桶)按 FIPS 从 soil_dict 查取;
    土壤仅作为静态上下文,不参与网格注意力计算。
    entry 为 None、l_enc==0 或缺土壤的样本会被跳过。
    """
    if dynamic_feature_names is None:
        dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)

    sub_samples = []
    skipped_soil = 0
    for meta, entry in pairs:
        if entry is None or int(entry["l_enc"]) == 0:
            continue
        fips = str(meta.get("FIPS", "")).zfill(5)
        srec = soil_dict.get(fips)
        if srec is None or any(srec.get(var) is None for var in SOIL_FEATURES):
            skipped_soil += 1
            continue
        soil_vec = torch.tensor([float(srec[var]) for var in SOIL_FEATURES], dtype=torch.float32)
        feats = entry["feats"]                   # (G, T, F) f32
        constructed = [n for n in CONSTRUCTED_FEATURES if n in dynamic_feature_names]
        if constructed and feats.shape[-1] < len(dynamic_feature_names):
            feats = append_constructed_features(feats, coords=entry["coords"], names=constructed)
        sub_samples.append({
            "grid_feats": feats,        # (G, T, F) f32
            "grid_coords": entry["coords"],      # (G, 2) f32 [lat, lon]
            "month": entry["month"],             # (T,) long
            "soil_feats": soil_vec,              # (7,) f32 连续土壤静态特征
            "yield_per_acre": torch.tensor([float(meta["yield_per_acre"])], dtype=torch.float32),
            "seq_len": int(entry["l_enc"]),
            "state": str(meta["State"]),
            "year": int(meta["Year"]),
            "FIPS": fips,
            "County": str(meta.get("County", "")),
        })
    if skipped_soil:
        print(f"  [提示] 跳过 {skipped_soil} 个缺土壤样本")
    return sub_samples


class GridTimeSeriesDataset(Dataset):
    """
    网格级时序数据集。

    每个样本一条 (县, 年):
      - grid_feats: (G, T, F) 该县覆盖 G 个 9×9km 网格的气象时序
      - grid_coords: (G, 2) 每个网格的 [lat, lon]
      - month: (T,) 时间步月份 (3-11)
      - static_bucket_ids: {"carbon_bucket": (1,), "ph_bucket": (1,)}
      - yield_per_acre: (1,) 单产 (bu/ac)
      - seq_len: 有效时间步数
    """

    def __init__(self, sub_samples: List[Dict]):
        self.sub_samples = sub_samples

    def __len__(self) -> int:
        return len(self.sub_samples)

    def __getitem__(self, idx: int):
        s = self.sub_samples[idx]
        return (
            s["grid_feats"],        # (G, T, F)
            s["grid_coords"],       # (G, 2)
            s["month"],             # (T,)
            s["soil_feats"],        # (7,) 连续土壤静态特征
            s["yield_per_acre"],    # (1,)
            s["seq_len"],
            s["state"],
            s["year"],
            s["FIPS"],
            s["County"],
        )


def make_grid_collate_fn(
    global_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    dynamic_feature_names: Optional[List[str]] = None,
    eps: float = 1e-6,
):
    """
    网格级 collate:标准化 + 填充 G/T + grid_mask + 土壤静态特征标准化。

    global_stats 需含 "dynamic" 与 "soil" 键,各为 (mean, std)。
    网格坐标 [lat,lon] 保持原始度数——位置编码在模型内用正余弦公式
    (π/180 转弧度)处理,无需外部统计量。
    返回元组:
      grid_feats (B,Gmax,Tmax,F), grid_coords (B,Gmax,2), grid_mask (B,Gmax) bool,
      month (B,Tmax) long, soil_feats (B,7), labels (B,1), seq_lens (B,),
      states, years, fips_list, county_list
    """
    mean_d, std_d = global_stats["dynamic"]
    mean_s, std_s = global_stats["soil"]
    feat_names = list(dynamic_feature_names or DEFAULT_DYNAMIC_FEATURE_NAMES)
    F = len(feat_names)

    def collate_fn(batch):
        grid_feats_list = [it[0] for it in batch]
        grid_coords_list = [it[1] for it in batch]
        month_list = [it[2] for it in batch]
        soil_list = [it[3] for it in batch]
        labels = [it[4] for it in batch]
        seq_lens = torch.tensor([int(it[5]) for it in batch], dtype=torch.long)
        states = [it[6] for it in batch]
        years = [int(it[7]) for it in batch]
        fips_list = [it[8] for it in batch]
        county_list = [it[9] for it in batch]

        B = len(batch)
        Gmax = max(x.shape[0] for x in grid_feats_list)
        Tmax = int(seq_lens.max().item())

        grid_feats = torch.zeros(B, Gmax, Tmax, F, dtype=torch.float32)
        grid_coords = torch.zeros(B, Gmax, 2, dtype=torch.float32)
        grid_mask = torch.zeros(B, Gmax, dtype=torch.bool)
        month_padded = torch.zeros(B, Tmax, dtype=torch.long)

        for i in range(B):
            gf = _apply_standardize(grid_feats_list[i], mean_d, std_d, eps)  # (G,T,F)
            g, t = gf.shape[0], gf.shape[1]
            grid_feats[i, :g, :t] = gf
            grid_coords[i, :g] = grid_coords_list[i]             # 坐标保持原始度数,正余弦编码在模型内处理
            grid_mask[i, :g] = True
            month_padded[i, :t] = month_list[i]

        # 土壤静态特征标准化 (B, 7),只作静态上下文,不进网格注意力
        soil_feats = torch.stack(
            [_apply_standardize(s, mean_s, std_s, eps).squeeze(0) for s in soil_list], dim=0
        )
        labels = torch.stack(labels, dim=0)

        return (
            grid_feats,      # (B, Gmax, Tmax, F)
            grid_coords,     # (B, Gmax, 2)
            grid_mask,       # (B, Gmax) bool
            month_padded,    # (B, Tmax) long
            soil_feats,      # (B, 7)
            labels,          # (B, 1)
            seq_lens,        # (B,)
            states, years, fips_list, county_list,
        )

    return collate_fn
