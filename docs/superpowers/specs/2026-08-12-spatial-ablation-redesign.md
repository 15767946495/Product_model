# 空间消融重构设计

## 目标

将当前消融实验收敛为三组纯空间对照，并修正网格坐标语义：

- A `mean`：有效网格算术平均，不使用位置编码；
- B `additive`：CLS cross-attention + 纯空间加性正余弦位置编码；
- C `rope`：CLS cross-attention + 纯空间二维 RoPE。

时间 RoPE 从模型、训练、推理和消融接口中完整删除。所有空间注意力均在同一时间步的网格之间计算，因此空间位置编码不得包含时间索引。

## 网格中心坐标

当前缓存使用原始 CSV 的 `Lat (llcrnr)` 和 `Lon (llcrnr)`，实际表示网格左下角。新缓存使用网格中心：

```text
lat_center = (Lat (llcrnr) + Lat (urcrnr)) / 2
lon_center = (Lon (llcrnr) + Lon (urcrnr)) / 2
```

`train_dataset/grid_cache.pt` 的版本从 2 升级为 3。删除旧缓存后，从原始 WRF-HRRR CSV 重新生成，禁止继续使用版本 2 的左下角坐标缓存。

## 空间聚合接口

模型使用显式 `spatial_encoding` 三态：

- `none`：仅用于 `spatial_mode=mean`；
- `additive`：空间加性正余弦编码；
- `rope`：二维空间 RoPE。

不再使用 `grid_rope` 布尔值隐式区分编码方式，也不提供旧 checkpoint 兼容逻辑。新实验使用新的输出 tag 和目录名，避免加载旧模型。

## CLS Cross-Attention

CLS 只作为县级聚合 query，不参与 key/value：

```text
query = W_q(CLS)
keys = W_k(grid_tokens)
values = W_v(grid_tokens)
county_feature = softmax(query @ keys) @ values
```

注意力只在真实有效网格上归一化，输出空间权重形状为 `(B, T, G)`。不存在 `CLS -> CLS` 自注意力。

对于 `additive`：

- 中心经纬度经纯空间正余弦编码后加到网格 token；
- 同一网格在全部时间步使用相同位置编码；
- CLS query 不加空间编码。

对于 `rope`：

- 中心经纬度仅旋转网格 key；
- CLS query 保持空间中性；
- 不编码时间轴。

## VSN 与特征结构

保留现有逐特征空间聚合。15 个动态特征分别生成县级时间序列，再由 TFT 的时序 VSN 做特征筛选，不把 15 个特征预先融合为一个 token。

## 消融脚本

`TFT_model/ablation_rope.py` 改为三组固定的 15 维构造特征实验：

| 编号 | 目录 | 空间模式 | 空间编码 |
|---|---|---|---|
| 0 | `mean` | mean | none |
| 1 | `additive` | attention | additive |
| 2 | `rope` | attention | rope |

脚本保留串行、`--combo`、`--force`、`--infer-only` 和并行能力。并行模式最多需要三张可见 GPU，按组合顺序一张卡运行一个实验。

## 验证

- 单元测试验证中心坐标是四角中点；
- 重建后验证缓存 `version == 3`；
- 抽样对比缓存坐标与原始 CSV 中心坐标；
- 测试 CLS 不出现在 key/value 中，空间权重最后一维等于真实网格维；
- 测试加性编码对不同时间步完全一致；
- 测试模型和 CLI 中不再存在时间 RoPE；
- 测试三组消融命名、参数和三卡映射；
- 运行模型前向、语法检查和 product 环境测试。

## 兼容性

该改动有意不兼容旧 `grid_cache.pt` 和旧 checkpoint。旧实验目录保留用于历史比较，但不得由新代码继续训练或推理。
