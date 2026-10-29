"""论文结果的唯一出处：滚动起点评价、置换检验、聚类自助法、季内截止分析与图表。

所有数值在这里一次性计算并写入 paper_figures/report.json 与 paper_figures/tables.md，论文正文只引用这些输出。
面板回归的每个预测年份只使用早于该年份的数据（forward_only=True）；2022 为最后一个年份，因此其结果与
"2017—2021 训练、2022 测试"的深度模型协议严格同口径。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from TFT_model.rs_signal_test import (
    _prepare_block, _ridge_by_year_cv, flatten, fold_predict, load, score, within, within_year_permutation,
)

OUT = _PROJECT_ROOT / "paper_figures"
RUNS = _PROJECT_ROOT / "TFT_model" / "output_dir" / "tft"
STATE_NAMES = {
    "05": "阿肯色", "08": "科罗拉多", "10": "特拉华", "13": "佐治亚",
    "16": "爱达荷", "17": "伊利诺伊", "18": "印第安纳", "19": "爱荷华",
    "20": "堪萨斯", "21": "肯塔基", "22": "路易斯安那", "24": "马里兰",
    "26": "密歇根", "27": "明尼苏达", "28": "密西西比", "29": "密苏里",
    "31": "内布拉斯加", "36": "纽约", "37": "北卡罗来纳", "38": "北达科他",
    "39": "俄亥俄", "40": "俄克拉何马", "42": "宾夕法尼亚", "45": "南卡罗来纳",
    "46": "南达科他", "47": "田纳西", "48": "得克萨斯", "51": "弗吉尼亚",
    "55": "威斯康星",
}
SETS = ("county_fe_only", "weather", "veg_indices", "pvt_frozen", "weather+veg", "weather+pvt")
LABELS = {"county_fe_only": "仅县固定效应", "weather": "县FE+天气", "veg_indices": "县FE+植被指数",
          "pvt_frozen": "县FE+冻结PVT", "weather+veg": "县FE+天气+植被指数", "weather+pvt": "县FE+天气+冻结PVT"}
TFT_GROUPS = {
    "TFT-PVT（原配置）": ["20261015-115150", "20261023-repro-main-s42", "20261023-main-s1", "20261023-main-s2",
                        "20261023-mon-main-s42", "20261023-mon-main-s1", "20261024-main-s3"],
    "TFT-数值遥感特征": [f"20261025-rsfeat-s{s}" for s in (42, 1, 2, 3, 4, 5)],
    "TFT-常数（零初始化）": ["20261018-remote-constant", "20261023-const-s1", "20261023-const-s2", "20261024-const-s3"],
    "TFT-常数（PVT均值初始化）": [f"20261024-constpm-s{s}" for s in (42, 1, 2, 3)],
}
BASELINES = {
    "UNet-ConvLSTM": "UNet-ConvLSTM/output_dir/unet_convlstm/20261015-021813",
    "PhenoYieldNet": "PhenoYieldNet/output_dir/phenoyieldnet/20261015-224930",
    "GNN-RNN": "GNN-RNN/output_dir/gnn_rnn/20261015-021725",
}


def blocks_for(data, name, months=6, n_pca=32):
    always = np.ones((len(data["z"]), months), dtype=bool)
    weather = (*flatten(data["weather"][:, :months], always), False, None)
    veg = (*flatten(data["veg"][:, :months], data["month_valid"][:, :months]), True, None)
    pvt = (*flatten(data["pvt"][:, :months], data["month_valid"][:, :months]), True, n_pca)
    return {"county_fe_only": [], "weather": [weather], "veg_indices": [veg], "pvt_frozen": [pvt],
            "weather+veg": [weather, veg], "weather+pvt": [weather, pvt]}[name]


def fold(data, blocks, year, **kwargs):
    test, pred, alpha = fold_predict(blocks, data["z"], data["fips"], data["year"], year,
                                     forward_only=True, **kwargs)
    _, base, _ = fold_predict([], data["z"], data["fips"], data["year"], year, forward_only=True)
    return np.where(test)[0], pred, base, alpha


def metrics(y, y_hat):
    e = y_hat - y
    sst = ((y - y.mean()) ** 2).sum()
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
            "r2": float(1 - (e ** 2).sum() / sst), "corr": float(np.corrcoef(y, y_hat)[0, 1]),
            "bias": float(e.mean())}


def evaluate_years(data, name, years, months=6):
    blocks = blocks_for(data, name, months)
    per_year, pooled_y, pooled_p, pooled_idx, pooled_base = {}, [], [], [], []
    for year in years:
        idx, pred, base, alpha = fold(data, blocks, year)
        y = np.exp(data["z"][idx])
        s = score(data["z"][idx], pred, data["fips"][idx], data["year"][idx], base)
        per_year[int(year)] = {**metrics(y, np.exp(pred)), "anom_corr": s.get("within_state_year_anom_corr"),
                               "alpha": alpha, "n": int(len(idx))}
        pooled_y.append(y); pooled_p.append(np.exp(pred)); pooled_idx.append(idx); pooled_base.append(base)
    idx = np.concatenate(pooled_idx)
    s = score(data["z"][idx], np.log(np.concatenate(pooled_p)), data["fips"][idx], data["year"][idx],
              np.concatenate(pooled_base))
    return {"per_year": per_year,
            "pooled": {**metrics(np.concatenate(pooled_y), np.concatenate(pooled_p)),
                       "anom_corr": s.get("within_state_year_anom_corr")}}


def permutation_test(data, name, years, n_perm, seed=0):
    """把遥感块在每个年份内跨县置换（训练与测试行同时置换），得到'遥感无县年信息'的零分布。"""
    blocks = blocks_for(data, name)
    cache, rng = {}, np.random.default_rng(seed)

    def pooled(perm):
        ys, ps, bs, ids = [], [], [], []
        for year in years:
            idx, pred, base, _ = fold(data, blocks, year, permutation=perm, cache=cache)
            ys.append(np.exp(data["z"][idx])); ps.append(np.exp(pred)); bs.append(base); ids.append(idx)
        idx = np.concatenate(ids)
        s = score(data["z"][idx], np.log(np.concatenate(ps)), data["fips"][idx], data["year"][idx], np.concatenate(bs))
        return metrics(np.concatenate(ys), np.concatenate(ps))["rmse"], s["within_state_year_anom_corr"]

    obs_rmse, obs_corr = pooled(None)
    null = np.array([pooled(within_year_permutation(data["year"], rng)) for _ in range(n_perm)])
    return {"years": list(years), "n_perm": n_perm, "observed_rmse": obs_rmse, "observed_anom_corr": obs_corr,
            "null_rmse_mean": float(null[:, 0].mean()), "null_rmse_q05": float(np.quantile(null[:, 0], .05)),
            "null_corr_mean": float(null[:, 1].mean()), "null_corr_q95": float(np.quantile(null[:, 1], .95)),
            "p_rmse": float((1 + (null[:, 0] <= obs_rmse).sum()) / (n_perm + 1)),
            "p_corr": float((1 + (null[:, 1] >= obs_corr).sum()) / (n_perm + 1))}


def load_predictions(run_dir):
    rows = [json.loads(line) for line in open(run_dir / "county_predictions.jsonl")]
    return {r["fips"]: (r["y_pred"], r["y_true"]) for r in rows}


def cluster_bootstrap(e1, e2, states, reps, rng, by_state=True):
    """RMSE(1) − RMSE(2) 的自助区间；by_state=True 按州整体重抽（保留州内空间相关）。"""
    n = len(e1)
    groups = [np.where(states == s)[0] for s in np.unique(states)]
    diffs = np.empty(reps)
    for b in range(reps):
        if by_state:
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        else:
            idx = rng.integers(0, n, n)
        diffs[b] = np.sqrt((e1[idx] ** 2).mean()) - np.sqrt((e2[idx] ** 2).mean())
    return float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975)), float((diffs < 0).mean())


def main(args):
    OUT.mkdir(exist_ok=True)
    data = load(args.features)
    previous = json.loads((OUT / "report.json").read_text(encoding="utf-8")) if args.reuse_permutations else None
    report, rng = {}, np.random.default_rng(0)
    roll_years = (2020, 2021, 2022)

    # 1) 滚动起点：逐年与合并
    report["rolling"] = {name: evaluate_years(data, name, roll_years) for name in SETS}
    for name in SETS:
        print(name, json.dumps({k: round(v, 3) for k, v in report["rolling"][name]["pooled"].items()}), flush=True)

    # 2) 2022 逐县预测、州分解、自助法
    idx22, pred22, base22, alpha22 = fold(data, blocks_for(data, "weather+veg"), 2022)
    fips22, y22 = data["fips"][idx22], np.exp(data["z"][idx22])
    _, w_pred, _, _ = fold(data, blocks_for(data, "weather"), 2022)
    _, fe_pred, _, _ = fold(data, [], 2022)
    states = np.array([f[:2] for f in fips22])
    report["state_rmse_2022"] = {
        STATE_NAMES[s]: {"n": int((states == s).sum()),
                         "weather+veg": float(np.sqrt(((np.exp(pred22) - y22)[states == s] ** 2).mean())),
                         "county_fe_only": float(np.sqrt(((np.exp(fe_pred) - y22)[states == s] ** 2).mean()))}
        for s in np.unique(states)}
    others = {"县FE+天气": np.exp(w_pred), "仅县固定效应": np.exp(fe_pred)}
    for name, path in BASELINES.items():
        pred = load_predictions(_PROJECT_ROOT / path)
        others[name] = np.array([pred[f][0] for f in fips22])
    tft = load_predictions(RUNS / TFT_GROUPS["TFT-PVT（原配置）"][0])
    others["TFT-PVT（原报告运行）"] = np.array([tft[f][0] for f in fips22])
    e_main = np.exp(pred22) - y22
    report["bootstrap_vs"] = {}
    for name, p in others.items():
        lo, hi, share = cluster_bootstrap(e_main, p - y22, states, args.boot, rng, by_state=True)
        lo_i, hi_i, share_i = cluster_bootstrap(e_main, p - y22, states, args.boot, rng, by_state=False)
        report["bootstrap_vs"][name] = {
            "rmse_other": metrics(y22, p)["rmse"], "diff_state_ci": [lo, hi], "share_neg_state": share,
            "diff_county_ci": [lo_i, hi_i], "share_neg_county": share_i}
    report["main_2022"] = {**metrics(y22, np.exp(pred22)), "alpha": alpha22, "n": int(len(idx22)),
                           "anom_corr": report["rolling"]["weather+veg"]["per_year"][2022]["anom_corr"]}

    # 3) 置换检验（耗时；--reuse-permutations 时沿用上一次 report.json 中的结果）
    if previous is not None:
        report["permutation_2022"] = previous["permutation_2022"]
        report["permutation_2020_2022"] = previous["permutation_2020_2022"]
    else:
        report["permutation_2022"] = {name: permutation_test(data, name, (2022,), args.perm)
                                      for name in ("veg_indices", "weather+veg", "pvt_frozen", "weather+pvt")}
        report["permutation_2020_2022"] = {name: permutation_test(data, name, roll_years, args.perm_pooled)
                                           for name in ("weather+veg", "weather+pvt")}
    for k in ("permutation_2022", "permutation_2020_2022"):
        for name, v in report[k].items():
            print(k, name, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}, flush=True)

    # 4) 季内截止
    report["cutoff"] = {}
    for months in range(1, 7):
        report["cutoff"][months + 3] = {
            name: evaluate_years(data, name, roll_years, months)
            for name in ("county_fe_only", "weather", "weather+veg")}

    # 5) 月份消融（探索性）
    veg_all = blocks_for(data, "weather+veg")
    weather_block, veg_block = veg_all
    X, valid, is_rs, _ = veg_block
    k = X.shape[1] // 6
    report["month_ablation"] = {}
    for m in range(6):
        sel = np.zeros(X.shape[1], bool); sel[m * k:(m + 1) * k] = True
        for tag, keep in (("only", sel), ("drop", ~sel)):
            blk = [weather_block, (X[:, keep], valid[:, keep], True, None)]
            ys, ps = [], []
            for year in roll_years:
                idx, pred, _, _ = fold(data, blk, year)
                ys.append(np.exp(data["z"][idx])); ps.append(np.exp(pred))
            report["month_ablation"][f"{tag}_{m + 4}"] = metrics(np.concatenate(ys), np.concatenate(ps))

    # 6) 系数（2022 折，标准化尺度）：按特征块与月份汇总 L2 范数
    train = data["year"] < 2022
    parts = [_prepare_block(Xb, vb, data["fips"], train, nb, 0) for Xb, vb, _, nb in veg_all]
    Xw = np.concatenate(parts, axis=1)
    z_mean = {f: data["z"][train & (data["fips"] == f)].mean() for f in np.unique(data["fips"][train])}
    base_all = np.array([z_mean.get(f, np.nan) for f in data["fips"]])
    model, alpha = _ridge_by_year_cv(Xw[train], (data["z"] - base_all)[train], data["year"][train])
    coef = model.coef_
    names_w = [str(n) for n in np.load(args.features)["weather_features"]]
    names_v = [str(n) for n in np.load(args.features)["veg_features"]]
    layout = [(f"天气:{n}", m) for m in range(6) for n in names_w] + [(f"植被:{n}", m) for m in range(6) for n in names_v]
    # _standardize 会丢弃训练折内方差为 0 的列；用同一规则还原保留列，使系数与特征名一一对应
    kept = np.concatenate([
        within(Xb, data["fips"], train, vb)[train].std(0) > 1e-10 for Xb, vb, _, _ in veg_all
    ])
    labels = [l for l, keep_flag in zip(layout, kept) if keep_flag]
    assert len(labels) == len(coef), (len(labels), len(coef))
    by_group = {}
    for (name, month), c in zip(labels, coef):
        kind = name.split(":")[0]
        by_group.setdefault(f"{kind}|{month + 4}", []).append(c * c)
    report["coef_norm_by_group"] = {k_: float(np.sqrt(sum(v))) for k_, v in by_group.items()}
    top = sorted(zip(labels, coef), key=lambda t: -abs(t[1]))[:12]
    report["top_coef"] = [{"feature": f"{n}@{m + 4}月", "coef": float(c)} for (n, m), c in top]
    report["alpha_final"] = alpha

    # 7) TFT 各配置汇总
    summary = {}
    for group, runs in TFT_GROUPS.items():
        rows = []
        for run in runs:
            t = json.load(open(RUNS / run / "test_metrics.json"))
            pred = np.array([v[0] for v in load_predictions(RUNS / run).values()])
            rows.append({"run": run, "rmse": t["rmse"], "mae": t["mae"], "r2": t["r2"], "corr": t["corr"],
                         "pred_sd": float(pred.std()), "dead": bool(pred.std() < 1.0)})
        r = np.array([x["rmse"] for x in rows]); live = np.array([x["rmse"] for x in rows if not x["dead"]])
        summary[group] = {"runs": rows, "n": len(rows), "mean": float(r.mean()), "sd": float(r.std(ddof=1)),
                          "median": float(np.median(r)), "min": float(r.min()), "max": float(r.max()),
                          "n_dead": int(sum(x["dead"] for x in rows)), "mean_live": float(live.mean()),
                          "sd_live": float(live.std(ddof=1))}
    report["tft_groups"] = summary
    report["baselines"] = {name: json.load(open(_PROJECT_ROOT / path / "test_metrics.json"))
                           for name, path in BASELINES.items()}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    report = json.loads(json.dumps(report))     # 键统一为字符串，与读取 report.json 的口径一致
    figures(data, report, idx22, pred22, y22, fips22)
    print("saved report.json")


def figures(data, report, idx22, pred22, y22, fips22):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from TFT_model.attention_analysis import _configure_chinese_font
    _configure_chinese_font()
    plt.rcParams["axes.unicode_minus"] = False

    # 图3-1 植被指数季节轨迹
    fig, ax = plt.subplots(figsize=(6.4, 4))
    exg = data["veg"][:, :, 0]
    months = np.arange(4, 10)
    for yr, col in zip(range(2017, 2023), plt.cm.viridis(np.linspace(0, .9, 6))):
        ax.plot(months, exg[data["year"] == yr].mean(0), marker="o", color=col, label=str(yr))
    ax.set_xlabel("月份"); ax.set_ylabel("县均值 ExG"); ax.legend(ncol=3, fontsize=8)
    ax.set_title("各年份超绿指数(ExG)的季节轨迹")
    fig.tight_layout(); fig.savefig(OUT / "fig3_1_exg_seasonal.png", dpi=200); plt.close(fig)

    # 图6-1 观测—预测散点
    fig, ax = plt.subplots(figsize=(5.2, 5))
    st = np.array([f[:2] for f in fips22])
    for s in np.unique(st):
        sel = st == s
        ax.scatter(y22[sel], np.exp(pred22)[sel], s=14, label=STATE_NAMES[s], alpha=.8)
    lim = [min(y22.min(), np.exp(pred22).min()) - 5, max(y22.max(), np.exp(pred22).max()) + 5]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("观测单产 (bu/ac)"); ax.set_ylabel("预测单产 (bu/ac)")
    ax.set_title(f"2022年县级预测（RMSE={report['main_2022']['rmse']:.2f}）")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(OUT / "fig7_1_scatter.png", dpi=200); plt.close(fig)

    # 图6-2 各配置 RMSE 分布
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    groups = list(report["tft_groups"])
    for i, g in enumerate(groups):
        rows = report["tft_groups"][g]["runs"]
        xs = i + (np.random.default_rng(0).random(len(rows)) - .5) * .25
        ax.scatter(xs, [r["rmse"] for r in rows], c=["tab:red" if r["dead"] else "tab:blue" for r in rows], s=22, zorder=3)
        ax.hlines(report["tft_groups"][g]["median"], i - .3, i + .3, color="k", lw=1.2, zorder=2)
    ax.axhline(report["main_2022"]["rmse"], color="tab:green", ls="--", label=f"县FE+天气+植被指数 ({report['main_2022']['rmse']:.2f})")
    fe = report["rolling"]["county_fe_only"]["per_year"]["2022"]["rmse"]
    ax.axhline(fe, color="gray", ls=":", label=f"仅县固定效应 ({fe:.2f})")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g.replace("（", "\n（") for g in groups], fontsize=8)
    ax.set_ylabel("2022年RMSE (bu/ac)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.set_ylim(top=27.5)
    ax.set_title("TFT各配置多次训练的RMSE（红点：预测退化为常数；黑线：中位数）")
    fig.tight_layout(); fig.savefig(OUT / "fig7_3_tft_seeds.png", dpi=200); plt.close(fig)

    # 图6-3 PVT 坍缩轨迹
    fig, ax = plt.subplots(figsize=(6.6, 4))
    for run, label in (("20261023-mon-main-s42", "原配置 种子42"), ("20261023-mon-main-s1", "原配置 种子1"),
                       ("20261023-mon-main-pvtlr0.1-s42", "PVT学习率×0.1"), ("20261023-mon-rsvar-s42", "遥感作为VSN变量"),
                       ("20261023-mon-rsaux-s42", "遥感辅助损失")):
        rows = [json.loads(l) for l in open(RUNS / run / "pvt_probe.jsonl")]
        ax.plot([r["epoch"] for r in rows], [r["tokens.pairwise_cos_mean"] for r in rows], marker=".", label=label)
    ax.axhline(.95, color="gray", ls=":"); ax.set_xlabel("训练轮次（-1为初始化）")
    ax.set_ylabel("PVT输出的平均两两余弦相似度"); ax.legend(fontsize=8)
    ax.set_title("PVT遥感表示的坍缩过程（训练年份子集）")
    fig.tight_layout(); fig.savefig(OUT / "fig7_4_pvt_collapse.png", dpi=200); plt.close(fig)

    # 图6-4 季内截止：逐折显示，避免只呈现最有利的2022折
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharex=True)
    ms = sorted(report["cutoff"], key=int)
    for ax, year in zip(axes, ("2020", "2021", "2022")):
        for name, col in (("county_fe_only", "gray"), ("weather", "tab:blue"), ("weather+veg", "tab:green")):
            ax.plot([int(m) for m in ms], [report["cutoff"][m][name]["per_year"][year]["rmse"] for m in ms],
                    marker="o", ms=4, color=col, label=LABELS[name])
        ax.set_title(f"{year}年折"); ax.set_xlabel("截止月份")
    axes[0].set_ylabel("RMSE (bu/ac)"); axes[0].legend(fontsize=7)
    fig.suptitle("季内截止分析（逐折）：训练仅用预测年份之前的数据", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig7_2_cutoff.png", dpi=200); plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser("Thesis numbers and figures")
    parser.add_argument("--features", default="/data/raid0/hqx/train_dataset/rs_signal_features.npz")
    parser.add_argument("--perm", type=int, default=1000)
    parser.add_argument("--perm-pooled", type=int, default=300)
    parser.add_argument("--boot", type=int, default=5000)
    parser.add_argument("--reuse-permutations", action="store_true",
                        help="reuse the permutation-test results already stored in paper_figures/report.json")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
