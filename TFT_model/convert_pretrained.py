"""将 MMST-ViT 预训练 checkpoint 的 PVT backbone 权重转换为 TFT 格式。
TFT 的 PVTTinyEncoder 前三 stage 与 MMST-ViT 的 pvt_tiny 完全等价，
仅 key 名和最后一 stage 输出维度不同（512 vs 128）。
"""
import argparse
import re
from pathlib import Path

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmst-checkpoint",
                        default="MMST-ViT/output_current/pretrain/checkpoint-latest.pth")
    parser.add_argument("--output", default="MMST-ViT/output_current/tft_pretrain.pth")
    args = parser.parse_args()

    src = torch.load(args.mmst_checkpoint, map_location="cpu", weights_only=False)
    src_state = src.get("model", src) if isinstance(src, dict) else src

    # ------ key 映射 ------
    mapped = {}
    skipped = []

    for key, value in src_state.items():
        if not key.startswith("backbone."):
            continue
        suffix = key[len("backbone."):]

        # stage 映射: patch_embedN -> stages.{N-1}
        match = re.match(r"^patch_embed(\d)\.(.+)$", suffix)
        if match:
            stage = int(match.group(1)) - 1
            new_key = f"pvt.stages.{stage}.{match.group(2)}"
        # block 映射: blockN.M -> blocks.{N-1}.{M}
        elif re.match(r"^block\d", suffix):
            new_key = re.sub(r"^block(\d)\.(\d)", lambda m: f"blocks.{int(m.group(1))-1}.{m.group(2)}", suffix)
            new_key = f"pvt.{new_key}"
        # pos_embed 映射: pos_embedN -> pos_embeds.{N-1}
        elif re.match(r"^pos_embed\d", suffix):
            new_key = re.sub(r"^pos_embed(\d)", lambda m: f"pos_embeds.{int(m.group(1))-1}", suffix)
            new_key = f"pvt.{new_key}"
        # cls_token -> cls_token
        elif suffix == "cls_token":
            new_key = f"pvt.cls_token"
        # norm -> norm
        elif suffix.startswith("norm.") or suffix == "norm":
            new_key = f"pvt.{suffix}"
        # head -> 跳过
        elif suffix.startswith("head."):
            skipped.append((key, "classification head"))
            continue
        else:
            skipped.append((key, "unmapped key"))
            continue

        mapped[new_key] = value

    # 构建输出 checkpoint
    payload = {"model": mapped}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"转换完成: {len(mapped)} 个参数迁移, {len(skipped)} 个跳过")
    print(f"输出: {args.output}")
    if skipped:
        print("跳过的参数 (前10条):")
        for k, reason in skipped[:10]:
            print(f"  {k}: {reason}")