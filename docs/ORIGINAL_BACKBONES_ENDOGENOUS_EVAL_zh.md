# 原始基座：全部变量训练，仅内生变量评估

## 命令指定 CUDA 设备

```bash
bash scripts/run_query_fusion.sh hipatch mimic original --device cuda:1
bash scripts/run_original_backbones.sh --device cuda:1
```

直接把 cuda:1 改为 cuda:0/cuda:2/cuda:3 即可，不改变 batch、模型或训练参数。
显式 --device 不再施加脚本默认的 CUDA_VISIBLE_DEVICES=0，也优先于旧 GPU 环境变量。
如果调用者/调度器已设置 CUDA_VISIBLE_DEVICES，则保留该限制，cuda:N 仍指可见设备中的编号；
未设置时与服务器物理编号对应。训练进程也将 CUDA 当前设备设置为所选卡。
DEVICE=cuda:1 的环境变量写法仍支持；旧 GPU=1 的重映射写法保持兼容。
同一参数也支持 Hi-Patch/mTAND base-mlp-v1 和 STRGCN 插件批量脚本。

复用当前仓库已有 t-PatchGNN、Hi-Patch 和 mTAN 官方网络实现，不接入 MLP 或注意力插件。
运行变体为 `original`，与 `base` 区分：`base` 是 Y-only 历史与 Y-only 训练；
`original` 保留所有原始变量的历史、预测头和训练目标。

## 仅在评估时选择变量

训练仍对全部有效未来观测计算原有变量宏平均 MSE；mTAN 仍加现有的 KL 项。
验证/测试先生成全部变量预测，再在指标 mask 中筛选内生变量：

`evaluation_mask = prediction_mask & padding_mask & is_endogenous(variable_id)`

不把这个评估 mask 传给模型，不修改训练 mask，不改变历史输入、原始列顺序或网络结构。
所有未来变量都是预测标签，均不作为模型输入，包括划为 X 的变量。
验证集 MSE、早停和最佳 checkpoint 选择均使用内生变量；训练目标保持全部变量。
因此最佳轮次可能与原来全变量评估时不同，这是评估范围变化的直接结果。

| 数据集 | 历史/预测头变量数 | 内生评估数 |
|---|---:|---:|
| Activity | 12 | 6 |
| MIMIC | 96 | 52 |
| PhysioNet | 41 | 35 |
| USHCN | 5 | 2 |

内生集合与当前角色文件及插件任务完全相同，MIMIC 的 52、76 两列不参与评估；
原始基座仍保留这两列的输入/输出容量与有效观测训练，与原模型的完整变量集合一致。
静态变量保留原始有效记录，不人为复制未来基线值。

## 其余实验设置

直接复用当前四数据集的记录划分、归一化、历史/预测区间和 Y 可评估窗口，
与现有 base/mlp/v1 对照对齐。该入口不是对原论文全部训练协议的重新复现；
mTAN 继续采用仓库现有的 forecasting 任务适配（MSE+KL），不是改回论文的 IWAE 目标。
基座超参数沿用各自当前配置，固定 seed 3407，最多30轮/5轮早停。

实现：`AllVariableForecastDataset.py` 只复制 packet 元数据，训练目标扩展为全部原始列；
`TPatchForecastTrainer.evaluate` 在预测完成后筛选指标。原始记录和 vendored 网络代码均不修改。
`evaluation_scope: all` 可恢复全变量评估，用于检查两种评估下输入、预测和训练损失相同。
旧的插件配置没有 evaluation_target_ids，因此保持现有评估结果不变。

导出的 test_y_pred/test_y_true 数组保留完整预测位置，test_y_mask 仅选择内生变量。
计算指标时必须使用该 mask。metrics.json/evaluation_protocol.json 分别记录 training_target_ids、
target_ids（评估集合）、原始输入协议；不会把全变量训练误写成 Y-only 训练。

## 一次运行三个原模型与四个数据集

```bash
bash scripts/run_original_backbones.sh
```

共12次顺序实验，每个组合只跑 seed3407，默认 GPU0，新建结果目录并拒绝覆盖旧结果。

只运行一个原模型的四个数据集：

```bash
bash scripts/run_original_backbones.sh hipatch
bash scripts/run_original_backbones.sh tpatchgnn
bash scripts/run_original_backbones.sh mtand
```

只运行一个原模型、一个数据集：

```bash
bash scripts/run_query_fusion.sh hipatch activity original
```

GPU、PYTHON、DEVICE、TPATCH_DATA_ROOT、RUN_DIR、DRY_RUN 环境变量与现有脚本一致。

## Hi-Patch / MIMIC 显存不足时

原始实现显式构造观测级关系矩阵和边消息，默认 batch=32 可能超出空闲48GB卡。
专用入口默认选择物理GPU1、batch=1，保持网络宽度、patch和内生评估不变：

```bash
GPU=1 bash scripts/run_hipatch_mimic_original.sh
```

等价的通用入口（现已支持 BATCH_SIZE）：

```bash
GPU=1 BATCH_SIZE=1 bash scripts/run_query_fusion.sh hipatch mimic original
```

GPU 指物理卡编号；CUDA_VISIBLE_DEVICES 重映射后，进程内仍使用 cuda:0，
因此报错中 GPU0 不代表选错物理卡。不要同时指定 DEVICE=cuda:1。
batch大小写入运行配置、manifest及metrics协议；改变batch会改变优化过程，比较时需注明。
只减小batch不能保证所有记录都适合显存，专用入口尚未在远程GPU上验证。

## 验证

2026-11-18：127项相关测试通过，其中新增15项；三个基座×四个真实数据集，
共12次CPU单轮、小模型、每划分最多2窗口的诊断训练均成功。
由导出的数组独立重算全部12次内生MSE/MAE/RMSE，与报告一致。
这不代表完成全量GPU训练或验证了精度提升。
详见 `reports/original_endogenous_validation.json`。
