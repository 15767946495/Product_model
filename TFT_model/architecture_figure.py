"""第4章图4-1：PAF-Net 模型架构。

输出 `paper_figures/fig4_1_architecture.png`（300 dpi）。
模块画成浅立体块；气象为日粒度（168 日），遥感为月粒度（6 个月）并前向
对齐到日；VSN 按日步展开，空间注意力展开为“网格块 → 加权聚合 → 县—日
向量”，LSTM 展开为若干时间步块。实线为前向计算，虚线为条件化与状态初始化。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"

FONT = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
FONT_NAME = font_manager.FontProperties(fname=FONT).get_name()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_NAME],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

INK = "#1E2833"
MUTED = "#5B6572"
ARROW = "#4A5563"

C_INPUT = ("#F3F4F6", "#9AA3AF")
C_ENC = ("#E8EEF4", "#7D93AB")
C_STATIC = ("#EDE9F6", "#8A79B0")
C_PANEL = ("#FBE7D8", "#C47A45")
C_PANEL_BG = ("#FDF6F0", "#E7C9B2")
C_TEMP = ("#DCE9F7", "#5F86B0")
C_TEMP_BG = ("#F4F8FD", "#C9D8E8")
C_GRID = ("#E7EEF5", "#8097AC")
C_OUT = ("#E4F0E8", "#6A9A7A")

FIG_W, FIG_H = 7.4, 7.1
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")


def shade(color, factor):
    """Lighten (factor > 0) or darken (factor < 0) a color."""
    r, g, b = to_rgb(color)
    if factor >= 0:
        return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)
    f = 1 + factor
    return (r * f, g * f, b * f)


def cuboid(x, y, w, h, colors, depth=0.12, layers=1):
    """Draw a shallow 3D block with a front face, top face and right face."""
    fc, ec = colors
    dx, dy = depth, depth * 0.75
    ax.add_patch(Polygon(
        [(x, y + h), (x + dx, y + h + dy),
         (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True, facecolor=shade(fc, 0.45), edgecolor=ec,
        linewidth=0.7, joinstyle="round", zorder=2))
    ax.add_patch(Polygon(
        [(x + w, y), (x + w + dx, y + dy),
         (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True, facecolor=shade(fc, -0.14), edgecolor=ec,
        linewidth=0.7, joinstyle="round", zorder=2))
    for k in range(1, layers):
        t = k / layers
        ax.plot([x + dx * t, x + w + dx * t],
                [y + h + dy * t, y + h + dy * t],
                color=shade(ec, 0.25), lw=0.35, alpha=0.75, zorder=2)
        ax.plot([x + w + dx * t, x + w + dx * t],
                [y + dy * t, y + h + dy * t],
                color=shade(ec, -0.05), lw=0.35, alpha=0.75, zorder=2)
    ax.add_patch(Rectangle(
        (x, y), w, h, facecolor=fc, edgecolor=ec,
        linewidth=0.9, zorder=3))


def card(x, y, w, h, title, body="", colors=C_INPUT, title_size=7.0,
         body_size=5.8, depth=0.12, layers=1, tag=None, tag_size=5.6):
    cuboid(x, y, w, h, colors, depth=depth, layers=layers)
    ax.text(x + w / 2, y + h - 0.09, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=INK, zorder=4)
    if body:
        ax.text(x + w / 2, y + h - 0.26, body, ha="center", va="top",
                fontsize=body_size, color=INK, zorder=4, linespacing=1.30)
    if tag:
        ax.text(x + w - 0.06, y + 0.04, tag, ha="right", va="bottom",
                fontsize=tag_size, color=MUTED, zorder=4)


def panel(x, y, w, h, colors):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=1))


def arrow(start, end, dashed=False, color=ARROW, lw=0.9, ms=6):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=ms,
                        linestyle=(0, (3, 2)) if dashed else "-",
                        shrinkA=0, shrinkB=0),
        zorder=5)


def route(pts, dashed=False, color=ARROW, lw=0.9, ms=6):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw,
            ls=(0, (3, 2)) if dashed else "-",
            solid_capstyle="round", zorder=3)
    ax.annotate(
        "", xy=pts[-1], xytext=pts[-2],
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=ms,
                        linestyle=(0, (3, 2)) if dashed else "-",
                        shrinkA=0, shrinkB=0),
        zorder=5)


# ---------------------------------------------------------------- 输入
card(0.25, 6.35, 0.95, 0.52, "县级土壤", r"$S_c$：7 维",
     title_size=6.8, body_size=5.6)
card(1.55, 6.35, 1.75, 0.52, "网格逐日气象",
     r"$W_{cygd}$：12 变量" "\n" r"× 168 日 × $G_c$ 网格")
card(3.65, 6.35, 1.75, 0.52, "网格—月植被指数",
     r"$V_{cymg}$：10 指数" "\n" r"× 6 月 × $G_c$ 网格")
card(5.75, 6.35, 1.20, 0.52, "网格坐标与日序",
     r"$\ell_{cg}$；$d=1,\ldots,168$", title_size=6.8, body_size=5.6)

# ------------------------------------------------------------ 特征编码
card(0.25, 5.58, 0.95, 0.52, "静态编码器", "四路上下文",
     C_STATIC, title_size=6.6, body_size=5.6)
card(1.55, 5.58, 1.75, 0.52, "气象特征",
     "训练集标准化\n" r"逐变量嵌入（$12\to H$）", C_ENC)
card(3.65, 5.58, 1.75, 0.52, "遥感特征",
     r"全局标准化，$10\to H$" "\n逐月嵌入", C_ENC)
card(5.75, 5.58, 1.20, 0.52, "位置编码", "网格坐标 + 日序", C_ENC,
     title_size=6.8, body_size=5.6)

# ------------------------------------------------------ 时间粒度条
n_days = 24
for i in range(n_days):
    x = 1.65 + i * (3.20 - 1.65) / (n_days - 1)
    ax.add_patch(Rectangle(
        (x - 0.013, 5.39), 0.026, 0.08,
        facecolor=C_ENC[1], edgecolor="none", alpha=0.75, zorder=4))
ax.text(1.60, 5.43, "气象：日粒度 168 步", ha="right", va="center",
        fontsize=5.2, color=MUTED, zorder=5)

for i in range(6):
    x = 3.80 + i * 0.26
    ax.add_patch(Rectangle(
        (x, 5.39), 0.18, 0.08, facecolor=C_ENC[0], edgecolor=C_ENC[1],
        linewidth=0.5, zorder=4))
    ax.text(x + 0.09, 5.43, str(4 + i), ha="center", va="center",
            fontsize=4.4, color=INK, zorder=5)
ax.text(5.35, 5.43, "遥感：月粒度 6 步", ha="left", va="center",
        fontsize=5.2, color=MUTED, zorder=5)

for x in (2.425, 4.525):
    arrow((x, 5.58), (x, 5.47), lw=0.8, ms=5)

# ------------------------------------------------------ 月 → 日前向对齐
cuboid(4.20, 4.98, 1.70, 0.30, C_PANEL, depth=0.08, layers=1)
ax.text(5.05, 5.22, "月→日前向对齐", ha="center", va="center",
        fontsize=5.6, fontweight="bold", color=INK, zorder=4)
ax.text(5.05, 5.07, "6 月 → 168 日", ha="center", va="center",
        fontsize=5.0, color=INK, zorder=4)
arrow((4.54, 5.39), (4.54, 5.28), lw=0.8, ms=5)
arrow((4.90, 4.98), (4.90, 4.68), lw=0.9)
ax.text(4.98, 4.83, "逐日展开", ha="left", va="center",
        fontsize=5.0, color=MUTED, zorder=5)

# 遥感的第二条路：与土壤一起条件化 VSN
route([(5.90, 5.13), (6.15, 5.13), (6.15, 4.20), (5.75, 4.20)],
      dashed=True, color=C_PANEL[1], lw=0.8)
ax.text(6.20, 4.55, "RS 条件", ha="left", va="center",
        fontsize=5.0, color=MUTED, zorder=5)

# ------------------------------------------------------ 逐日 VSN
ax.text(0.55, 4.90, "逐日网格—日变量选择（VSN）", ha="left", va="top",
        fontsize=6.8, fontweight="bold", color=INK, zorder=5)
panel(0.55, 3.70, 6.20, 1.05, C_PANEL_BG)

vsn_xs = [2.55, 3.40, 4.25, 5.10]
vsn_labels = [r"$d{=}1$", r"$d{=}2$", r"$\cdots$", r"$d{=}168$"]
for x, label in zip(vsn_xs, vsn_labels):
    cuboid(x, 3.90, 0.65, 0.55, C_PANEL, depth=0.08, layers=3)
    ax.text(x + 0.325, 4.175, label, ha="center", va="center",
            fontsize=5.4, color=INK, zorder=4)

# 日输入总线：气象与对齐后的遥感
ax.plot([2.875, 5.425], [4.68, 4.68], color=ARROW, lw=0.8, zorder=3)
for x in (2.875, 3.725, 4.575, 5.425):
    arrow((x, 4.68), (x, 4.45), lw=0.7, ms=4)
arrow((2.875, 5.39), (2.875, 4.68), lw=0.9)

# 土壤条件与遥感条件（共享到整组日步）
arrow((0.45, 4.20), (2.55, 4.20), dashed=True,
      color=C_STATIC[1], lw=0.8)
ax.text(0.60, 4.26, "土壤条件", ha="left", va="bottom",
        fontsize=5.0, color=MUTED, zorder=5)
ax.text(4.15, 3.82, r"每个日步 × $G_c$ 个网格", ha="center", va="top",
        fontsize=5.2, color=MUTED, zorder=5)

# VSN 输出进入空间注意力
arrow((3.40, 3.70), (3.40, 3.22), lw=0.9)
ax.text(3.48, 3.62, r"$\mathbf{x}_{cygd}$（逐日）", ha="left", va="center",
        fontsize=5.0, color=MUTED, zorder=5)

# ---------------------------------------------------- 空间注意力（展开）
panel(0.55, 2.55, 6.45, 1.00, C_PANEL_BG)
ax.text(0.70, 3.47, "逐日县内空间注意力", ha="left", va="top",
        fontsize=7.4, fontweight="bold", color=INK, zorder=5)
ax.text(0.70, 3.33, "CLS 查询 + 加权聚合", ha="left", va="top",
        fontsize=5.6, color=MUTED, zorder=5)

cube_xs = [0.90, 1.40, 1.90, 2.40, 2.90]
for x in cube_xs:
    cuboid(x, 2.85, 0.24, 0.24, C_GRID, depth=0.07, layers=1)
ax.text(1.90, 2.74, r"$G_c$ 个网格", ha="center", va="top",
        fontsize=5.4, color=MUTED, zorder=5)

# VSN 输出进入每个网格
ax.text(3.48, 3.18, "", ha="left", va="bottom")
ax.plot([0.90, 3.40], [3.22, 3.22], color=ARROW, lw=0.8, zorder=3)
for x in cube_xs:
    arrow((x + 0.12, 3.22), (x + 0.12, 3.09), lw=0.7, ms=4)

# 网格 → 注意力加权 → 县—日向量
weights = [0.55, 0.95, 1.35, 0.80, 0.60]
for x, wt in zip(cube_xs, weights):
    ax.add_patch(FancyArrowPatch(
        (x + 0.12, 2.85), (3.90, 2.97),
        arrowstyle="-|>", mutation_scale=5,
        lw=0.50 + 0.55 * wt, color=C_PANEL[1], alpha=0.72,
        shrinkA=0, shrinkB=0, zorder=4))
ax.add_patch(plt.Circle((3.96, 2.97), 0.065, facecolor="white",
                        edgecolor=C_PANEL[1], lw=0.8, zorder=5))
ax.text(3.96, 2.97, "∑", ha="center", va="center",
        fontsize=6.5, color=C_PANEL[1], zorder=6)

cuboid(4.22, 2.81, 0.35, 0.32, C_PANEL, depth=0.10, layers=1)
ax.text(4.84, 2.97,
        r"$\mathbf{v}_{cyd}=\sum_{g=1}^{G_c}\alpha_{cygd}\,\mathbf{x}_{cygd}\in\mathbb{R}^{H}$",
        ha="left", va="center", fontsize=5.8, color=INK, zorder=5)
arrow((4.03, 2.97), (4.22, 2.97), color=C_PANEL[1], lw=0.8)

# 县—日向量接入 LSTM 的逐日输入总线
arrow((4.395, 2.81), (4.395, 1.92), color=ARROW, lw=0.9)
ax.text(4.34, 2.40, r"逐日 $d=1,\ldots,168$", ha="right", va="center",
        fontsize=5.0, color=MUTED, zorder=5)

# --------------------------------------------------------- LSTM（展开）
panel(0.55, 1.15, 3.75, 1.00, C_TEMP_BG)
ax.text(0.70, 2.08, "单向 LSTM（168 日步）", ha="left", va="top",
        fontsize=7.0, fontweight="bold", color=INK, zorder=5)

ax.plot([0.85, 4.395], [1.92, 1.92], color=ARROW, lw=0.8, zorder=3)

lstm_xs = [0.85, 1.65, 2.45, 3.25]
lstm_labels = [r"$d{=}1$", r"$d{=}2$", r"$\cdots$", r"$d{=}168$"]
for x, label in zip(lstm_xs, lstm_labels):
    cuboid(x, 1.35, 0.55, 0.36, C_TEMP, depth=0.08, layers=1)
    ax.text(x + 0.275, 1.53, label, ha="center", va="center",
            fontsize=5.6, color=INK, zorder=4)
    arrow((x + 0.275, 1.92), (x + 0.275, 1.71), lw=0.7, ms=4)

for x in lstm_xs[:-1]:
    arrow((x + 0.63, 1.53), (x + 0.80, 1.53), lw=0.8, ms=5)
ax.text(1.90, 1.25, r"$h_{d-1}\rightarrow h_d$", ha="center", va="top",
        fontsize=5.0, color=MUTED, zorder=5)

# 静态状态初始化
arrow((0.45, 1.53), (0.85, 1.53), dashed=True, lw=0.8)
ax.text(0.60, 1.62, r"$h_0,c_0$", ha="left", va="bottom",
        fontsize=5.0, color=MUTED, zorder=5)

# ------------------------------------------------------ 时间注意力
panel(4.60, 1.15, 2.45, 1.00, C_TEMP_BG)
ax.text(4.72, 2.08, "未来屏蔽注意力", ha="left", va="top",
        fontsize=7.0, fontweight="bold", color=INK, zorder=5)
card(4.85, 1.35, 1.15, 0.50, "多头注意力", "因果掩码",
     C_TEMP, title_size=6.4, body_size=5.4, depth=0.10, layers=3)

# 因果掩码小图标
for i in range(4):
    for j in range(4):
        if j <= i:
            ax.add_patch(Rectangle(
                (6.14 + j * 0.05, 1.82 - i * 0.05), 0.040, 0.040,
                facecolor=C_TEMP[1], edgecolor="none", alpha=0.75, zorder=4))
ax.text(6.14, 1.60, r"$t\leq d$", ha="left", va="top",
        fontsize=5.0, color=MUTED, zorder=5)

arrow((3.88, 1.53), (4.85, 1.53), lw=0.9)
arrow((5.425, 1.35), (5.425, 0.95), lw=0.9)
ax.text(5.48, 1.10, r"$\hat z_{cyd}$", ha="left", va="bottom",
        fontsize=5.0, color=MUTED, zorder=5)

# ------------------------------------------------------------ 预测输出
card(4.85, 0.25, 1.10, 0.70, "预测头（GRN）",
     r"逐日输出 $\hat z_{cyd}$", C_TEMP,
     title_size=6.6, body_size=5.4, depth=0.12, layers=3)
card(6.15, 0.25, 1.00, 0.70, "输出",
     r"$\hat z_{cy}$：最后有效日"
     "\n" r"MSE vs $Z_{cy}$"
     "\n" r"$\hat Y=\exp(\hat z_{cy})$",
     C_OUT, title_size=6.6, body_size=4.9, depth=0.12, layers=3)
arrow((5.95, 0.60), (6.15, 0.60), lw=0.9)

# ------------------------------------------------------------ 输入箭头
for x in (0.725, 2.425, 4.525, 6.35):
    arrow((x, 6.35), (x, 6.10))

# 位置编码旁路进入空间注意力
route([(6.88, 5.58), (6.88, 3.55)], lw=0.9)

# 静态上下文总线
route([(0.45, 5.58), (0.45, 1.53)], dashed=True, lw=0.8)

# ------------------------------------------------------------------ 图例
ax.text(0.60, 0.60,
        "实线：前向计算\n虚线：静态条件与状态初始化\n"
        r"$G_c$：县内有效网格数；$H$：隐藏维度",
        fontsize=5.6, color=MUTED, ha="left", va="center", linespacing=1.45)

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig4_1_architecture.png", dpi=300,
            bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(OUT / "fig4_1_architecture.png")
