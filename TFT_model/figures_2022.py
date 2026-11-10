"""第5章图：2022 折（2017—2020 训练、固定轮数、2022 测试）。

图5-1  2022年观测—预测散点（PA-TFT 与仅县固定效应）
图5-2  分州RMSE哑铃图（PA-TFT 对仅县固定效应，按差值排序）
图5-3  面板设计交叉与组件消融的RMSE

预测来自 `/data/raid0/hqx/TFT_out/tft/diag2020-improved-y2022-s0`（环境变量
`LOYO_RUN`、`LOYO_FEATURES` 可覆盖）；仅县固定效应与图5-3 的数值口径与
`paper_report_tft.py`、`single_fold_report.py` 一致，分别读取特征文件的训练年份县均值
与 `paper_figures/single_fold_2022.json`。
写入 `paper_figures/fig5_1_scatter.png`、`fig5_2_state_rmse.png`、`fig5_3_ablation.png`。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFT_model.attention_analysis import _configure_chinese_font  # noqa: E402
from TFT_model.paper_report import STATE_NAMES, metrics  # noqa: E402

OUT = ROOT / "paper_figures"
RUN = Path(os.environ.get("LOYO_RUN", "/data/raid0/hqx/TFT_out/tft/diag2020-improved-y2022-s0"))
FEATURES = os.environ.get("LOYO_FEATURES", "/data/raid0/hqx/train_dataset/rs_signal_features_31states.npz")
TRAIN_LAST_YEAR = 2020

# 分类色板（dataviz 校验通过：蓝 #2a78d6 / 橙 #eb6834）
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_DARK = "#1c5cab"
INK, MUTED, CONNECTOR, GRID = "#0b0b0b", "#52514e", "#c9c8c3", "#e4e3de"


def load_run(run_dir: Path):
    rows = sorted((json.loads(line) for line in open(run_dir / "county_predictions.jsonl")),
                  key=lambda r: r["fips"])
    return (np.array([r["fips"] for r in rows]),
            np.array([r["y_pred"] for r in rows]),
            np.array([r["y_true"] for r in rows]))


def county_feats():
    """训练年份县均值；返回 (county_mean, state_name_lookup)。"""
    from TFT_model.rs_signal_test import load as load_features
    data = load_features(FEATURES)
    train = data["year"] <= TRAIN_LAST_YEAR
    return {c: data["z"][train & (data["fips"] == c)].mean() for c in np.unique(data["fips"])}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, width=0.8)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def fig_scatter(plt, fips, pred, truth, fe):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7), constrained_layout=True)
    lim = (0.0, 260.0)
    panels = (("（a）PA-TFT", pred, BLUE), ("（b）仅县固定效应", fe, ORANGE))
    for ax, (title, p, color) in zip(axes, panels):
        m = metrics(truth, p)
        ax.plot(lim, lim, ls="--", lw=1, color=MUTED, zorder=1)
        ax.scatter(truth, p, s=13, color=color, alpha=0.55, linewidths=0, zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(f"{title}（RMSE = {m['rmse']:.2f}）", fontsize=11, color=INK)
        ax.set_xlabel("观测单产（bu/ac）", fontsize=10, color=INK)
        ax.text(0.05, 0.95, f"RMSE {m['rmse']:.2f}\nMAE {m['mae']:.2f}\n$R^2$ {m['r2']:.3f}\n$r$ {m['corr']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK,
                bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID, "lw": 0.8})
        style(ax)
    axes[0].set_ylabel("预测单产（bu/ac）", fontsize=10, color=INK)
    fig.savefig(OUT / "fig5_1_scatter.png", dpi=200)
    plt.close(fig)
    return {t: metrics(truth, p) for (t, p, _) in panels}


def fig_state_dumbbell(plt, fips, pred, truth, fe):
    states = np.array([f[:2] for f in fips])
    rows = []
    for s in np.unique(states):
        sel = states == s
        rows.append({"name": STATE_NAMES[s], "n": int(sel.sum()),
                     "tft": float(np.sqrt(((pred[sel] - truth[sel]) ** 2).mean())),
                     "fe": float(np.sqrt(((fe[sel] - truth[sel]) ** 2).mean()))})
    rows.sort(key=lambda r: r["tft"] - r["fe"])
    fig, ax = plt.subplots(figsize=(6.8, 7.6), constrained_layout=True)
    ys = np.arange(len(rows))
    for y, r in zip(ys, reversed(rows)):
        ax.plot([r["fe"], r["tft"]], [y, y], color=CONNECTOR, lw=1.6, zorder=1, solid_capstyle="round")
    ax.scatter([r["fe"] for r in reversed(rows)], ys, s=30, color=ORANGE, zorder=3, label="仅县固定效应")
    ax.scatter([r["tft"] for r in reversed(rows)], ys, s=30, color=BLUE, zorder=3, label="PA-TFT")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['name']}（{r['n']}）" for r in reversed(rows)], fontsize=8.5, color=INK)
    ax.set_xlabel("2022年分州RMSE（bu/ac）", fontsize=10, color=INK)
    ax.set_xlim(0, 45)
    style(ax)
    for side in ("left",):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False, axis="y")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    gains = [r["fe"] - r["tft"] for r in rows]
    better = int(sum(g > 0 for g in gains))
    median_gain = float(np.median([g for g in gains if g > 0]))
    ax.set_title(f"{better}/{len(rows)}个州RMSE下降，中位降幅 {median_gain:.2f} bu/ac", fontsize=10, color=INK)
    fig.savefig(OUT / "fig5_2_state_rmse.png", dpi=200)
    plt.close(fig)
    return rows


def fig_ablation(plt, fe_rmse):
    report = json.loads((OUT / "single_fold_2022.json").read_text(encoding="utf-8"))
    cfgs = report["configs"]
    order = [
        ("improved", "完整PA-TFT"),
        ("wdemean", "气象也做县内去均值"),
        ("nors", "去掉遥感输入"),
        ("nodemean", "去掉植被指数县内去均值"),
        ("level", "水平目标（两项设计均不加）"),
        ("noffset", "去掉县均值偏移"),
    ]
    rows = [(label, cfgs[key]["rmse"]) for key, label in order]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    ys = np.arange(len(rows))
    colors = [BLUE_DARK if label == "完整PA-TFT" else BLUE for label, _ in rows]
    ax.barh(ys, [v for _, v in rows], height=0.62, color=colors, zorder=3)
    for y, (label, v) in zip(ys, rows):
        ax.text(v + 0.25, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK,
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"}, zorder=5)
    ax.axvline(fe_rmse, color=MUTED, ls=":", lw=1.2, zorder=2)
    ax.annotate(f"仅县固定效应 {fe_rmse:.2f}", xy=(fe_rmse, -0.62), xytext=(fe_rmse - 0.3, -0.62),
                ha="right", va="center", fontsize=8.5, color=MUTED, annotation_clip=False)
    ax.set_ylim(-1.0, len(rows) - 0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _ in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 26.5)
    ax.set_xlabel("2022年RMSE（bu/ac）", fontsize=10, color=INK)
    style(ax)
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "fig5_3_ablation.png", dpi=200)
    plt.close(fig)
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(exist_ok=True)

    fips, pred, truth = load_run(RUN)
    means = county_feats()
    fe = np.exp(np.array([means[c] for c in fips]))

    scores = fig_scatter(plt, fips, pred, truth, fe)
    states = fig_state_dumbbell(plt, fips, pred, truth, fe)
    fe_rmse = float(np.sqrt(((fe - truth) ** 2).mean()))
    ablation = fig_ablation(plt, fe_rmse)

    for name, m in scores.items():
        print(f"{name}: RMSE {m['rmse']:.2f} MAE {m['mae']:.2f} R2 {m['r2']:.3f} r {m['corr']:.3f}")
    gains = [r["fe"] - r["tft"] for r in states]
    print(f"states: {int(sum(g > 0 for g in gains))}/{len(gains)} better, "
          f"median gain {np.median([g for g in gains if g > 0]):.2f}")
    for label, v in ablation:
        print(f"{label}: {v:.2f} (+{v - 17.73:.2f})")
    print("wrote fig5_1_scatter.png, fig5_2_state_rmse.png, fig5_3_ablation.png")


if __name__ == "__main__":
    main()
