"""训练期 PVT 遥感分支监控：梯度、参数漂移、token 坍缩与预测敏感性。

逐步统计写入 <run_dir>/pvt_monitor.jsonl，逐 epoch 探针写入 <run_dir>/pvt_probe.jsonl。
监控只读取梯度与激活，不改变参数更新；探针只使用训练年份样本，在 eval 模式下运行。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from TFT_model import dataset_adapter as adapter
from TFT_model import rs_intervention as rsi
from TFT_model.models import RS_VARIABLE_NAME
from train_dataset import run_output as ro

PVT_PREFIX = "vit_encoder.backbone."
# 遥感 token 进入主干后的直接消费模块
DOWNSTREAM_PREFIXES = {
    "static_context_grn": "static_context_grn.",
    "vsn_context": "grid_vsn.flattened_grn.context.",
    "vsn_rs_grn": f"grid_vsn.single_var_grns.{RS_VARIABLE_NAME}.",
    "rs_aux_head": "rs_aux_head.",
}


def pvt_part(name: str) -> str:
    """把 backbone 参数名归到 stage / 组件。"""
    parts = name[len(PVT_PREFIX):].split(".")
    if parts[0] == "stages":
        return f"patch_embed{parts[1]}"
    if parts[0] == "blocks":
        return f"blocks{parts[1]}"
    return {"pos_embeds": "pos_embed", "cls_token": "cls_token", "norm": "final_norm"}.get(parts[0], "other")


def _norm(tensors) -> float:
    norms = [torch.linalg.vector_norm(t.detach().float()) for t in tensors if t is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0


def _specific_share(x: torch.Tensor) -> float:
    """样本特有能量占比：1 表示各样本互不相同，0 表示全部相同。"""
    deviation = x - x.mean(dim=0)
    return float(deviation.pow(2).sum() / x.pow(2).sum().clamp_min(1e-30))


def build_probe_loader(dataset, stats, *, samples: int, seed: int, use_constructed: bool,
                       batch_size: int = 4, num_workers: int = 2):
    """训练年份中固定的 samples 个县年，用独立随机数生成器抽取，不消耗全局随机状态。"""
    generator = torch.Generator().manual_seed(seed + 9973)
    indices = torch.randperm(len(dataset), generator=generator)[:samples].tolist()
    return adapter.build_loader(
        torch.utils.data.Subset(dataset, sorted(indices)), stats, batch_size, False,
        use_constructed=use_constructed, num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )


class PVTMonitor:
    def __init__(self, model, run_dir, *, every: int, probe_loader=None, device="cuda"):
        if getattr(model, "use_remote_constant", False):
            raise ValueError("PVT monitoring requires the PVT branch; remote_constant has none")
        self.model = model
        self.every = max(int(every), 1)
        self.probe_loader = probe_loader
        self.device = device
        self.step_path = Path(run_dir) / "pvt_monitor.jsonl"
        self.probe_path = Path(run_dir) / "pvt_probe.jsonl"
        self.step = 0
        self._capture = False
        self._pre_norm = []

        self.groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {}
        for name, parameter in model.named_parameters():
            if name.startswith(PVT_PREFIX):
                group = f"pvt.{pvt_part(name)}"
            else:
                group = next(
                    (label for label, prefix in DOWNSTREAM_PREFIXES.items() if name.startswith(prefix)),
                    "main_other",
                )
            self.groups.setdefault(group, []).append((name, parameter))
        self.initial = {
            name: parameter.detach().clone()
            for group, members in self.groups.items() if group != "main_other"
            for name, parameter in members
        }
        self._hook = model.vit_encoder.backbone.norm.register_forward_hook(self._capture_pre_norm)

    def _capture_pre_norm(self, _module, args, _output):
        if self._capture:
            self._pre_norm.append(args[0][:, 0].detach().float())

    def close(self):
        self._hook.remove()

    def before_forward(self):
        self._capture = self.step % self.every == 0
        self._pre_norm = []

    def after_backward(self, *, epoch: int, lr: float, loss: float, aux: dict, inputs: dict,
                       aux_loss: float | None = None):
        """在 optimizer.step() 之前调用：梯度已就绪、参数尚未更新。"""
        step, self.step = self.step, self.step + 1
        self._capture = False
        if step % self.every != 0:
            return
        row = {"epoch": epoch, "step": step, "lr": lr, "loss": loss}
        if aux_loss is not None:
            row["rs_aux_loss"] = aux_loss
        with torch.no_grad():
            for group, members in self.groups.items():
                row[f"grad_norm.{group}"] = _norm(p.grad for _, p in members)
                if group == "main_other":
                    continue
                weight = _norm(p for _, p in members)
                drift = _norm(p.detach() - self.initial[n] for n, p in members)
                row[f"weight_norm.{group}"] = weight
                row[f"rel_drift.{group}"] = drift / max(_norm(self.initial[n] for n, _ in members), 1e-12)
            row["grad_norm.pvt_total"] = _norm(
                p.grad for group, members in self.groups.items()
                if group.startswith("pvt.") for _, p in members
            )
            row.update(self._token_stats(aux, inputs))
            row.update(self._pre_norm_stats())
        self._pre_norm = []
        ro.append_jsonl(self.step_path, row)

    def _token_stats(self, aux: dict, inputs: dict) -> dict:
        rs = aux["grad_tensors"]["rs_encoded"]
        valid = rsi.valid_token_mask(inputs["grid_mask"], inputs["ag_grid_mask"])
        tokens = rs.detach()[valid].float()
        if tokens.shape[0] < 2:
            return {}
        deviation = tokens - tokens.mean(dim=0)
        stats = {
            "tokens.n": int(tokens.shape[0]),
            "tokens.mean_norm": float(tokens.mean(dim=0).norm()),
            "tokens.dev_to_mean": float(deviation.pow(2).sum(dim=1).mean().sqrt() / tokens.mean(dim=0).norm().clamp_min(1e-12)),
        }
        if rs.grad is None:
            return stats
        grad = rs.grad[valid].float()
        grad_deviation = grad - grad.mean(dim=0)
        stats["tokens.grad_norm"] = float(grad.norm())
        # 梯度中随样本变化的部分；接近 0 表示下游只关心遥感 token 的公共均值
        stats["tokens.grad_specific_share"] = float(grad_deviation.pow(2).sum() / grad.pow(2).sum().clamp_min(1e-30))
        # >0：梯度下降会缩小 token 相对均值的偏离（推动坍缩）；<0：推动分散
        stats["tokens.collapse_alignment"] = float(
            (grad_deviation * deviation).sum()
            / (grad_deviation.norm() * deviation.norm()).clamp_min(1e-30)
        )
        return stats

    def _pre_norm_stats(self) -> dict:
        if not self._pre_norm:
            return {}
        cls = torch.cat(self._pre_norm)
        if cls.shape[0] < 2:
            return {}
        norm = self.model.vit_encoder.backbone.norm
        normalized = torch.nn.functional.layer_norm(cls, cls.shape[-1:], eps=norm.eps)
        output = normalized * norm.weight.float() + norm.bias.float()
        return {
            # 末层 LayerNorm 之前 / 归一化后（无仿射）/ 输出的样本特有能量占比
            "cls.pre_norm_specific_share": _specific_share(cls),
            "cls.normalized_specific_share": _specific_share(normalized),
            "cls.output_specific_share": _specific_share(output),
            "final_norm.gamma_abs_mean": float(norm.weight.abs().mean()),
            "final_norm.beta_norm": float(norm.bias.norm()),
        }

    @torch.no_grad()
    def probe(self, epoch: int):
        if self.probe_loader is None:
            return
        self._capture = False
        was_training = self.model.training
        self.model.eval()
        try:
            collected = rsi.collect(self.model, self.probe_loader, self.device)
            labels, base = np.asarray(collected.labels), np.asarray(collected.preds)
            tokens, months = rsi.flatten_tokens(collected)
            shuffled = rsi.replay(self.model, collected, rsi.shuffle_within_month(collected, 0), self.device)
            constant = rsi.replay(
                self.model, collected, rsi.fill_month_vectors(collected, rsi.month_means(collected)), self.device,
            )
        finally:
            self.model.train(was_training)
        row = {
            "epoch": epoch,
            "step": self.step,
            **{f"tokens.{k}": v for k, v in rsi.token_statistics(tokens, months).items()},
            "train_subset_rmse": float(np.sqrt(np.mean((base - labels) ** 2))),
            "shuffle_mean_abs_delta_bu": float(np.abs(shuffled - base).mean()),
            "month_mean_mean_abs_delta_bu": float(np.abs(constant - base).mean()),
        }
        if collected.rs_vsn_weight:
            row["rs_vsn_weight_mean"] = float(np.mean(collected.rs_vsn_weight))
        ro.append_jsonl(self.probe_path, row)