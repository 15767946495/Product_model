"""图5-4：PAF-Net在2022年生长季内各月末截点的预测精度。

读取 paper_figures/inseason_2022.json（由 inseason_report.py 生成），
左图给出截断预测的RMSE、MAE随截点推进的变化，
右图给出州内异常相关与预测标准差。
写入 paper_figures/fig5_4_inseason.png。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFT_model.attention_analysis import _configure_chinese_font  # noqa: E402

OUT = ROOT / "paper_figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, width=0.8)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_chinese_font()
    plt.rcParams["axes.unicode_minus"] = False

    report = json.loads((OUT / "inseason_2022.json").read_text(encoding="utf-8"))
    keys = sorted(report["cutoffs"], key=int)
    rows = [report["cutoffs"][k] for k in keys]
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)

    ax = axes[0]
    rmse = [r["rmse"] for r in rows]
    mae = [r["mae"] for r in rows]
    ax.plot(x, rmse, "-o", color=BLUE, lw=1.8, ms=5, label="RMSE", zorder=3)
    ax.plot(x, mae, "--s", color=ORANGE, lw=1.5, ms=4.5, label="MAE", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("输入截断时点", fontsize=10, color=INK)
    ax.set_ylabel("误差（bu/ac）", fontsize=10, color=INK)
    ax.set_ylim(min(mae) - 6, max(rmse) + 4)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title("（a）整体精度", fontsize=11, color=INK)
    style(ax)

    ax = axes[1]
    anom = [r["anom_corr"] for r in rows]
    sd = [r["pred_sd"] for r in rows]
    ax.plot(x, anom, "-o", color=BLUE, lw=1.8, ms=5, label="州内异常相关", zorder=3)
    for xi, v in zip(x, anom):
        below = xi >= len(rows) - 2
        left = (not below) and 0 < xi < len(rows) - 1 and v < 0
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                    xytext=(0, -15) if below else ((-10, 9) if left else (0, 8)),
                    ha="right" if left else "center", fontsize=8, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("输入截断时点", fontsize=10, color=INK)
    ax.set_ylabel("州内异常相关", fontsize=10, color=INK)
    ax.set_ylim(min(anom) - 0.07, max(anom) + 0.07)
    style(ax)

    ax2 = ax.twinx()
    ax2.plot(x, sd, "--^", color=ORANGE, lw=1.4, ms=4, label="预测标准差", zorder=3)
    ax2.set_ylabel("预测标准差（bu/ac）", fontsize=10, color=ORANGE)
    ax2.tick_params(colors=ORANGE, labelsize=9)
    for side in ("top",):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_color(ORANGE)
    ax2.spines["right"].set_linewidth(0.8)
    ax.set_title("（b）县内排序与预测离散度", fontsize=11, color=INK)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=9, frameon=False, loc="lower right")

    fig.savefig(OUT / "fig5_4_inseason.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig5_4_inseason.png'}")
    for r in rows:
        print(f"{r['label']:>6s}: RMSE {r['rmse']:.2f} MAE {r['mae']:.2f} anom {r['anom_corr']:.3f} "
              f"sd {r['pred_sd']:.2f}")


if __name__ == "__main__":
    main()
