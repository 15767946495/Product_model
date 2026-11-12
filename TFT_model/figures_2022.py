"""第5章图：2022 折（2017—2021 训练、固定轮数、2022 测试）。

图5-1  2022年观测—预测散点（PA-TFT 与 GNN-RNN，后者是三个深度基线中最好的一组）
图5-2  分州RMSE哑铃图（PA-TFT 对 GNN-RNN，按差值排序）
图5-3  组件消融的RMSE

预测来自 `/data/raid0/hqx/TFT_out/tft/paper-main-y2022-s0`（环境变量 `LOYO_RUN`
可覆盖）；比较模型取 `paper_report_tft.py` 中 BASELINES 的同一运行目录。
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
RUN = Path(os.environ.get("LOYO_RUN", "/data/raid0/hqx/TFT_out/tft/paper-main-y2022-s0"))
BASELINE_RUN = Path(os.environ.get("LOYO_BASELINE_RUN", "/data/raid0/hqx/baselines/gnn_rnn/diag-gnn-vi-e100"))
BASELINE_LABEL = "GNN-RNN"

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


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, width=0.8)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def fig_scatter(plt, fips, pred, truth, base):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7), constrained_layout=True)
    lim = (0.0, 260.0)
    panels = (("（a）PA-TFT", pred, BLUE), (f"（b）{BASELINE_LABEL}", base, ORANGE))
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


def fig_state_dumbbell(plt, fips, pred, truth, base):
    states = np.array([f[:2] for f in fips])
    rows = []
    for s in np.unique(states):
        sel = states == s
        rows.append({"name": STATE_NAMES[s], "n": int(sel.sum()),
                     "tft": float(np.sqrt(((pred[sel] - truth[sel]) ** 2).mean())),
                     "base": float(np.sqrt(((base[sel] - truth[sel]) ** 2).mean()))})
    rows.sort(key=lambda r: r["tft"] - r["base"])
    fig, ax = plt.subplots(figsize=(6.8, 7.6), constrained_layout=True)
    ys = np.arange(len(rows))
    for y, r in zip(ys, reversed(rows)):
        ax.plot([r["base"], r["tft"]], [y, y], color=CONNECTOR, lw=1.6, zorder=1, solid_capstyle="round")
    ax.scatter([r["base"] for r in reversed(rows)], ys, s=30, color=ORANGE, zorder=3, label=BASELINE_LABEL)
    ax.scatter([r["tft"] for r in reversed(rows)], ys, s=30, color=BLUE, zorder=3, label="PA-TFT")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['name']}（{r['n']}）" for r in reversed(rows)], fontsize=8.5, color=INK)
    ax.set_xlabel("2022年分州RMSE（bu/ac）", fontsize=10, color=INK)
    ax.set_xlim(0, 80)
    style(ax)
    for side in ("left",):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False, axis="y")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    gains = [r["base"] - r["tft"] for r in rows]
    better = int(sum(g > 0 for g in gains))
    median_gain = float(np.median([g for g in gains if g > 0]))
    ax.set_title(f"{better}/{len(rows)}个州低于{BASELINE_LABEL}，中位降幅 {median_gain:.2f} bu/ac", fontsize=10, color=INK)
    fig.savefig(OUT / "fig5_2_state_rmse.png", dpi=200)
    plt.close(fig)
    return rows


def fig_ablation(plt, main_rmse):
    report = json.loads((OUT / "single_fold_2022.json").read_text(encoding="utf-8"))
    rows = [("PA-TFT（主模型）", report["main"]["rmse"])]
    rows += [(row["label"], row["rmse"]) for row in report.get("removal_ablation", {}).values()]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.0, 3.1), constrained_layout=True)
    ys = np.arange(len(rows))
    colors = [BLUE_DARK if label == "PA-TFT（主模型）" else BLUE for label, _ in rows]
    ax.barh(ys, [v for _, v in rows], height=0.62, color=colors, zorder=3)
    for y, (label, v) in zip(ys, rows):
        ax.text(v + 0.25, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK,
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"}, zorder=5)
    ax.set_ylim(-0.8, len(rows) - 0.35)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _ in rows], fontsize=9, color=INK)
    ax.set_xlim(0, max(v for _, v in rows) * 1.15)
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
    b_fips, base, b_truth = load_run(BASELINE_RUN)
    if not (b_fips == fips).all() or not np.allclose(b_truth, truth):
        raise SystemExit(f"baseline {BASELINE_RUN} is not on the same 2022 test set")

    scores = fig_scatter(plt, fips, pred, truth, base)
    states = fig_state_dumbbell(plt, fips, pred, truth, base)
    ablation = fig_ablation(plt, scores["（a）PA-TFT"]["rmse"])

    for name, m in scores.items():
        print(f"{name}: RMSE {m['rmse']:.2f} MAE {m['mae']:.2f} R2 {m['r2']:.3f} r {m['corr']:.3f}")
    gains = [r["base"] - r["tft"] for r in states]
    print(f"states: {int(sum(g > 0 for g in gains))}/{len(gains)} better, "
          f"median gain {np.median([g for g in gains if g > 0]):.2f}")
    for label, v in ablation:
        print(f"{label}: {v:.2f} ({v - scores['（a）PA-TFT']['rmse']:+.2f})")
    print("wrote fig5_1_scatter.png, fig5_2_state_rmse.png, fig5_3_ablation.png")


if __name__ == "__main__":
    main()
