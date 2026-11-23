"""图5-4：PAF-Net在2022年生长季内各月末截点的预测精度。

读取 paper_figures/inseason_2022.json（由 inseason_report.py 生成），
左图给出截断预测的RMSE随截点推进的变化，
右图给出Pearson相关系数随截点的变化。
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
BLUE = "#2a78d6"
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
    ax.plot(x, rmse, "-o", color=BLUE, lw=1.8, ms=5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("输入截断时点", fontsize=10, color=INK)
    ax.set_ylabel("RMSE（bu/ac）", fontsize=10, color=INK)
    ax.set_ylim(min(rmse) - 6, max(rmse) + 4)
    ax.set_title("（a）整体精度", fontsize=11, color=INK)
    style(ax)

    ax = axes[1]
    corr = [r["r"] for r in rows]
    ax.plot(x, corr, "-o", color=BLUE, lw=1.8, ms=5, zorder=3)
    for xi, v in zip(x, corr):
        if xi == len(rows) - 1:
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(-9, -13), ha="right", fontsize=8, color=BLUE)
        elif xi >= len(rows) - 2 or v < 0.2:
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=8, color=BLUE)
        else:
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(-10, 8), ha="right", fontsize=8, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("输入截断时点", fontsize=10, color=INK)
    ax.set_ylabel("Pearson相关系数", fontsize=10, color=INK)
    ax.set_ylim(min(corr) - 0.12, max(corr) + 0.10)
    ax.set_title("（b）相关系数", fontsize=11, color=INK)
    style(ax)

    fig.savefig(OUT / "fig5_4_inseason.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig5_4_inseason.png'}")
    for r in rows:
        print(f"{r['label']:>6s}: RMSE {r['rmse']:.2f} r {r['r']:.3f}")


if __name__ == "__main__":
    main()
