# WeatherFormer 时空加性注意力设计

## 目标

将当前空间聚合实验收敛为两组：

- `mean`：有效网格算术平均，不使用位置编码；
- `additive`：query-only CLS cross-attention，CLS 与网格 token 使用同一套 WeatherFormer 时空加性位置编码。

删除空间 RoPE、纯空间二维 additive 模式及其兼容接口。新的 attention 需要让每个时间步的 CLS query 不同，并显式建模相对日序、网格中心纬度和网格中心经度。

## 时间与空间坐标

时间使用序列相对日序：

```text
t = 0, 1, ..., T - 1
```

网格空间坐标继续使用缓存中的网格中心 `(lat_g, lon_g)`。

CLS 的空间坐标由每个样本内有效网格中心坐标的掩码均值计算：

```text
lat_cls = sum(mask_g * lat_g) / sum(mask_g)
lon_cls = sum(mask_g * lon_g) / sum(mask_g)
```

填充网格不得参与 CLS 坐标均值。数据契约保证每个有效样本至少包含一个有效网格；模型仍需使用分母下限避免除零。

## WeatherFormer 四槽编码

位置编码严格采用现有项目旧版 WeatherFormer 四槽布局。`hidden_size=d` 必须能被 4 整除。对 `i = 0, ..., d/4 - 1`：

```text
freq_i   = 10000^(-4i/d)
PE[4i]   = sin(t * freq_i)
PE[4i+1] = cos(t * freq_i)
PE[4i+2] = sin(lat_rad * freq_i)
PE[4i+3] = cos(lon_rad * freq_i)
```

其中 `lat_rad` 和 `lon_rad` 由角度乘 `pi/180` 得到。四槽含义为：

- 时间占两个槽位：`sin(t)` 与 `cos(t)`；
- 纬度占一个槽位：`sin(lat)`；
- 经度占一个槽位：`cos(lon)`。

CLS 和网格 token 必须调用同一个编码函数，区别仅在空间坐标：CLS 使用县内有效网格中心均值，网格 token 使用各自中心坐标。

## Query-Only CLS Cross-Attention

对每个动态特征分别执行网格聚合。给定内容 token `token_tg`：

```text
pe_cls_t  = PE(t, lat_cls, lon_cls)
pe_grid_tg = PE(t, lat_g, lon_g)

query_t = W_q(CLS + pe_cls_t)
key_tg  = W_k(token_tg + pe_grid_tg)
value_tg = W_v(token_tg)

weight_tg = softmax_g(query_t @ key_tg / sqrt(d))
county_feature_t = sum_g(weight_tg * value_tg)
```

设计约束：

- CLS 只作为 query，不进入 key/value；
- CLS query 与 grid key 使用统一的三坐标四槽编码；
- value 保持纯气象内容，不加入位置编码，避免位置直接污染县级气象表征；
- softmax 只在有效网格上归一化；
- 空间权重形状保持 `(B, T, G)`；
- 15 个动态特征继续分别聚合，之后由时序 VSN 做特征选择。

## 模型与接口

`SpatialAttentionAggregator` 不再接受 `spatial_encoding`。创建该模块即表示使用 WeatherFormer 时空加性编码。

`TFTEncoderForYieldPrediction` 保留 `spatial_mode`：

- `mean`：掩码网格均值；
- `attention`：WeatherFormer 时空加性 query-only cross-attention。

删除以下内容：

- `_rotate_half_pairwise`；
- `_apply_rotary`；
- `_grid_axis_boundaries`；
- `_grid_rope_cos_sin`；
- `spatial_encoding` 模型参数、CLI 参数和 checkpoint 字段；
- `rope` 消融组合及相关测试。

模型契约版本升级。旧 checkpoint 和旧 `model_hparams.json` 明确拒绝加载，不添加兼容分支。

## 训练与消融

消融脚本只运行两组固定的 15 维构造特征实验：

| 编号 | 目录 | 空间模式 | 位置编码 |
|---|---|---|---|
| 0 | `mean` | `mean` | 无 |
| 1 | `additive` | `attention` | WeatherFormer 时空加性编码 |

脚本保留串行、`--combo`、`--force`、`--infer-only` 和并行运行能力。并行运行两组时只要求两张可见 GPU。新实验使用新的输出 tag，当前及历史训练目录、checkpoint 不兼容且可直接删除。

## 验证

单元测试覆盖：

- 四槽公式及频率阶梯与 WeatherFormer 定义一致；
- `hidden_size` 不能被 4 整除时拒绝创建 attention 聚合器；
- 改变相对日序会同时改变 CLS 编码和网格编码；
- CLS 经纬度只由有效网格中心坐标计算；
- 填充网格不参与 CLS 坐标均值或 softmax；
- CLS 不进入 key/value，权重形状为 `(B,T,G)`；
- value 路径不加入位置编码；
- 模型、训练、推理和消融接口中不再存在 RoPE 或 `spatial_encoding`；
- 消融组合严格为 `mean` 和 `additive`，并行模式要求一组一张 GPU；
- mean 与 attention 的真实前向、语法检查和现有测试通过。

## 兼容性

该设计有意不兼容当前空间 2D additive/RoPE checkpoint。网格缓存坐标语义不变，继续使用版本 3 的网格中心坐标缓存，因此不需要重建 `grid_cache.pt`。
