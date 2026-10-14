"""统一回归指标：RMSE、MAE、标准 R2 = 1-SSE/SST、Pearson Corr、Corr2。"""

from __future__ import annotations

import math

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")
    if y_true.size == 0:
        raise ValueError("empty arrays")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("inputs contain NaN or Inf")
    error = y_pred - y_true
    sse = float(np.sum(error ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    if sst == 0:
        raise ValueError("R2 undefined: y_true is constant")
    if y_true.std() == 0 or y_pred.std() == 0:
        corr = 0.0  # 常量预测时相关系数无定义，按协议记为 0
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "rmse": math.sqrt(sse / y_true.size),
        "mae": float(np.mean(np.abs(error))),
        "r2": 1.0 - sse / sst,
        "corr": corr,
        "corr2": corr ** 2,
        "count": int(y_true.size),
    }
