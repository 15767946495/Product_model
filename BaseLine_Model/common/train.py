"""BaseLine_Model 通用训练循环。

各基线模型只需提供:
  - make_batches()   -> 每 epoch 的 batch 索引列表(生成器)
  - step_fn(model, opt, loss_fn, idx) -> 计算一个 batch 的 loss(含反向+更新),返回 loss.item()
  - eval_pred_fn(model) -> 全验证集的标准化预测 (Nval,)(模型输出需 squeeze)
训练/验证指标统一在原始 bu/ac 空间报告,与 TFT 口径一致。
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data import metrics, per_state_report, de_std


def run_training(model, data, device, args, tag, save_dir,
                 make_batches, step_fn, eval_pred_fn, test_pred_fn):
    tr, va, te = data["train"], data["val"], data["test"]
    stats = data["stats"]
    de = de_std(stats["ymean"], stats["ystd"])

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_rmse, best_state, patience = float("inf"), None, args.patience
    hist = {"epoch": [], "loss": [], "val_rmse": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        nb = 0
        for idx in make_batches():
            tot += step_fn(model, opt, loss_fn, idx)
            nb += 1

        model.eval()
        with torch.no_grad():
            pred_raw = de(np.asarray(eval_pred_fn(model)))
        m = metrics(pred_raw, np.asarray(va["y"]))
        hist["epoch"].append(epoch)
        hist["loss"].append(tot / nb)
        hist["val_rmse"].append(m["rmse"])

        if m["rmse"] < best_rmse - 1e-6:
            best_rmse = m["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d} loss={tot/nb:.4f} valRMSE={m['rmse']:.3f}")
        if patience <= 0:
            print(f"    early stop @ {epoch}")
            break

    # ---------- 最终评估(原始 bu/ac) ----------
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_raw = de(np.asarray(eval_pred_fn(model)))
    overall = metrics(pred_raw, np.asarray(va["y"]))
    per_state = per_state_report(va["meta"], pred_raw, np.asarray(va["y"]))
    with torch.no_grad():
        test_pred_raw = de(np.asarray(test_pred_fn(model)))
    test_overall = metrics(test_pred_raw, np.asarray(te["y"]))
    test_per_state = per_state_report(te["meta"], test_pred_raw, np.asarray(te["y"]))

    print(f"  ==== {tag}: RMSE={overall['rmse']:.3f} R²={overall['r2']:.3f} "
          f"Corr={overall['corr']:.3f} n={overall['n']}")
    for st, mm in per_state.items():
        print(f"    {st:12s} n={mm['n']:3d} RMSE={mm['rmse']:.3f} R²={mm['r2']:.3f}")
    print(f"  ==== {tag} TEST: RMSE={test_overall['rmse']:.3f} "
          f"R²={test_overall['r2']:.3f} Corr={test_overall['corr']:.3f} n={test_overall['n']}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_dir / f"best_{tag}.pth")
    result = {"model": tag, "validation": overall, "validation_per_state": per_state,
              "test": test_overall, "test_per_state": test_per_state,
              "epochs": len(hist["epoch"])}
    with open(save_dir / f"results_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(hist["epoch"], hist["loss"], color="#1f77b4", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train MSE", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist["val_rmse"], color="#d62728", label="val RMSE")
    ax2.set_ylabel("val RMSE (bu/ac)", color="#d62728")
    plt.title(tag)
    fig.tight_layout()
    fig.savefig(save_dir / f"curve_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"  已保存: {save_dir / ('best_' + tag + '.pth')}, "
          f"{save_dir / ('results_' + tag + '.json')}")
    return overall
