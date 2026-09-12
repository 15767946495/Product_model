# CropNet 八州四月至九月统一协议设计

## 目标

统一 MMST-ViT、TFT、CNN-RNN、ConvLSTM、GNN-RNN、DeepCropNet 和消融实验的数据协议，并清理旧协议模型权重与缓存。

本次只修改数据管线、配置和旧产物，不重新训练模型。

## 州范围

代码中使用显式允许州集合，只保留以下八州：

```text
Minnesota
Wisconsin
Michigan
Illinois
Indiana
Ohio
Missouri
Kentucky
```

代码中不出现被排除州的名称或缩写。通过允许列表自然排除其他州。

MMST-ViT 生成以下文件名：

```text
valid-no-ia.jsonl
valid-no-ia.train.jsonl
valid-no-ia.val.jsonl
valid-no-ia.test.jsonl

train.official.no-ia.json
val.official.no-ia.json
test.official.no-ia.json
```

上述文件名是既有运行协议的一部分，不能改成其他名称。

## 时间协议

所有气象模型统一使用官方 MMST-ViT 短期输入协议：

- 月份：4 月至 9 月；
- 每月日期：1 日至 28 日；
- 最大时间步：168；
- 计算方式：6 个月乘以每月 28 天；
- 遥感日期：每月第一幅影像，即 4 月 1 日至 9 月 1 日；
- Sentinel-2 文件：只引用 4 月至 6 月和 7 月至 9 月两个季度文件。

不再使用 3 月、10 月或 11 月数据，也不再使用旧的 275 步时间协议。

## 数据生成

### 县级 JSONL

`train_dataset/prepare_jsonl.py` 只处理允许州，并将天气记录限制到 4--9 月每月前 28 天。输出的每条样本使用新的时间范围。

### 网格缓存

`train_dataset/prepare_grid.py` 基于新的 JSONL 重建 `grid_cache.pt`。缓存版本递增，明确记录时间范围、最大时间步和允许州协议，防止旧缓存被误用。

### MMST-ViT manifest

`mmst_vit/manifest.py` 生成八州有效样本，并写出指定的 `valid-no-ia.*.jsonl` 文件。

`mmst_vit/config.py` 生成官方 JSON 数组，确保：

- 三个 split 均只包含允许州；
- 短期 HRRR 只包含 4--9 月；
- 长期 HRRR 保持官方前五年月度上下文格式；
- Sentinel-2 只包含 AG 和 NDVI 的两个生长季季度文件；
- USDA、HRRR 和 Sentinel 路径都可以被 `root_dir` 解析。

## 模型管线修改

### TFT

- 默认允许州集合改为八州；
- 读取新建的 168 步 JSONL 和网格缓存；
- 农学构造特征从 4 月 1 日开始累计；
- 训练与推理不再依赖 11 月 30 日锚点；
- 推理末端锚定到 9 月 28 日；
- 提前预报节点改为 4--9 月协议内的日期；
- 旧的 275 步、3--11 月和 11 月节点说明全部移除。

### CNN-RNN、ConvLSTM、GNN-RNN

共享数据模块统一为：

- 八州允许列表；
- `N_STEPS = 168`；
- 县均值、网格和图节点输入全部使用 168 步。

各模型同步更新输入形状、池化说明、注释和帮助文本。

### DeepCropNet

- 使用新的 4--9 月天气序列；
- 4 月 1 日作为序列起点；
- 保留论文中的 20 周、GDD/KDD/降水特征形式，但输入来源必须是新的 168 步序列；
- 区域映射只覆盖允许州。

### 消融实验

消融脚本使用相同的八州和 168 步数据源，不能自行读取旧数据缓存或旧州列表。

## 旧产物清理

删除 `/data/raid0/hqx` 下旧协议生成的模型权重，包括 TFT、CNN-RNN、ConvLSTM、GNN-RNN 和 DeepCropNet 权重。

删除会被新协议复用的旧数据缓存：

- 旧 `dataset.jsonl`；
- 旧 `grid_cache.pt`；
- 旧 `grid_cache_meta.json`；
- 旧基线数据缓存；
- 旧基线训练输出目录中的协议相关 checkpoint。

历史指标、曲线和日志不作为新实验输入；若保留，应与新协议产物分离。

不删除 Sentinel 原始下载文件，尤其不删除与其他州相关的可用遥感数据。

## 验收标准

### 代码协议

- 项目代码中的州允许列表只包含八州；
- 项目代码中不存在被排除州名称或缩写；
- 不存在旧的 275 步默认配置；
- 不存在 3--11 月或 11 月 30 日作为当前默认协议的逻辑。

### 数据协议

- 新 JSONL 不包含被排除州；
- 所有动态气象月份属于 4--9 月；
- 所有动态气象日期不超过当月 28 日；
- 网格缓存与 JSONL 行数一致；
- 新缓存的最大时间步为 168；
- MMST-ViT 三个 official JSON 不包含被排除州；
- MMST-ViT official JSON 只引用 4--9 月短期 HRRR 和两个 Sentinel 季度文件。

### 旧产物

- 旧 checkpoint 已删除；
- 旧缓存不会被默认训练入口加载；
- 新生成目录和旧日志目录清晰分离。

### 验证命令

至少运行：

```bash
python -m py_compile train_dataset/prepare_jsonl.py train_dataset/prepare_grid.py
python -m py_compile TFT_model/data.py TFT_model/train.py TFT_model/infer.py
python -m py_compile BaseLine_Model/common/data.py BaseLine_Model/common/train.py
python -m py_compile BaseLine_Model/CNNRNN/cnn_rnn.py
python -m py_compile BaseLine_Model/ConvLSTM/convlstm.py
python -m py_compile BaseLine_Model/GNNRNN/gnn_rnn.py
python -m py_compile BaseLine_Model/deepcropnet/deepcropnet.py
python -m py_compile mmst_vit/manifest.py mmst_vit/config.py
```

并使用 `hqx` 环境运行数据清单和缓存验证，不启动模型训练。

## 非目标

- 不修改 MMST-ViT、TFT 或基线模型的核心网络结构；
- 不重新训练任何模型；
- 不修复或重建已经下载的其他州 Sentinel 原始文件；
- 不将 3--9 月作为备用协议；
- 不在代码中加入被排除州名称或缩写。
