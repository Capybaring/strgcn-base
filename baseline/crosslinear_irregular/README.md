# CrossLinear 不规则预测基线

本目录提供 **CrossLinear+FutureX-MLP**：保留官方 CrossLinear 原模型，并增加 DAG 论文
附录 A.3 的未来外生融合方案，使 FullExo 接收历史 Y、历史 X 与已知未来 X。

模型来自 https://github.com/mumiao2000/CrossLinear ，精确提交记录于
`vendor/PROVENANCE.json`。`vendor/model.py` 与该版本 `models/CrossLinear.py` 字节完全一致，
附上游 MIT LICENSE。原始跨相关卷积、可学习 alpha/beta、patch 编码、位置嵌入、
预测头和内部归一化均保持原样。

单内生目标使用官方 `features=MS`，将目标排至最后一通道；多目标使用官方 `features=M`，
输入顺序为 `[X, Y]`，预测后取末尾 Y 通道，并保持配置中目标变量的原始顺序。
多目标模式使用官方完整多变量计算，不构造多个独立单目标模型；仅真实 Y 查询参与监督。

原始 CrossLinear 不接收未来 X，所以在原预测之后增加：

```
Linear(concat(Y_grid_prediction, X_future_grid), hidden=64)
→ ReLU → Linear(hidden, target_count)
```

这是按时间点进行的融合，所有预测时刻共用权重，不是残差相加。
新增融合层与原模型端到端共同训练；报告明确标记 `CrossLinear+FutureX-MLP`，不能称为完全未修改的原模型。
参考实现：`decisionintelligence/DAG@0758990` 的 `baselines/utils.py::MLP`。

在项目根目录安装与运行：

```bash
python -m pip install -r baseline/crosslinear_irregular/requirements.txt
python -m baseline.crosslinear_irregular.run \
  --config config/ExEn/processed/DIST.yaml \
  --mode FullExo --device cuda:0 --epochs 100 --batch-size 16 \
  --output results/baseline/CrossLinear/DIST_FullExo
```

全部数据集可使用 `bash baseline/run_full_exo.sh`，同时执行 TimeXer 和 CrossLinear 的 FullExo。
共享协议见 [公共说明](../common/README.md)。模型参数可放在数据 YAML 顶层：

```yaml
crosslinear:
  seq_len: 96
  patch_len: 16
  d_model: 512
  d_ff: 2048
  alpha: 1.0
  beta: 1.0
  fusion_hidden: 64
```

采用官方 MSE 目标、Adam lr=.0001、type1 学习率日程。这里最大 epoch=100、早停 patience=10、
batch=16 是本项目统一的运行默认值，并非声称复现论文全部调参结果；可用命令行覆盖。
NoExo 删除真实 X，并关闭整个未来 X 融合层；仅保留固定零兼容输入通道与历史 Y。
小规模验证加 `--tiny --epochs 2 --max-windows 2 --batch-size 2`。
输出目录必须为空。验证摘要见 `validation.json`；不以小规模训练结果比较性能。
