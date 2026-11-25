# 同维密度直接拼接验证（2026-11-25）

新增 `voronoi_h_4d` 和 `voronoi_h_4d_no_density`，
公式及运行入口见 [同维编码说明](../docs/VORONOI_CONCAT_ENCODING_zh.md)。
直接在原 master 工作，旧版本名称、检查点含义和历史结果保留。

## 实现边界

- 数学密度不变：内部 $\rho_i=2/(\delta_i^L+\delta_i^R)$，首尾对称延拓，
  $s_i=\log(\mu\rho_i)$，单独一个时刻记零。
- 无偏置 `Linear(1,d)` 把 s 编码成 d 维；值、变量、时间、密度四路各 d 维，拼接为 4d。
- 第一层直接使用 `LayerNorm(4d)` 和 Q/K/V `Linear(4d,d)`。
  拼接后没有融合 MLP；残差支路用无偏置 `Linear(4d,d)` 对齐，后续图层仍为 d 维。
- attention 后原有 FFN、时间/变量关系及 X→Y 融合保留，无独立密度 logit 偏置。
- 消融把整个 d 维密度段置零，保留同样的 4d 结构与参数数量。
- 六个 backbone 共享该插件；历史编码器、原预测头及各自训练损失保持原样。

## 自动测试

以下联合回归 **387 项全部通过**，CPU 耗时 96.45 秒：

```text
tests/test_voronoi_concat_encoding.py
tests/test_irregular_backbone_adapters.py
tests/test_voronoi_graph_transformer.py
tests/test_irregular_graph_transformer.py
tests/test_query_fusion.py
tests/test_strgcn_query_fusion.py
tests/test_mtan_query_fusion.py
tests/test_dist_query_fusion.py
tests/test_query_fusion_runner.py
tests/test_device_selection.py
```

新增/扩展验证覆盖同维四段输入、无融合 MLP、Q/K/V 与残差尺寸、密度编码器梯度、
规则采样精确等于同权重消融、消融密度段严格为零且不受其编码权重影响、
一层/两层图、分块 checkpoint 与完整计算的输出/梯度一致、无效 padding、token 排列、
无 X 恒等映射、未来 X 信息流、未来 Y 标签隔离，以及六模型/五数据集接口与反向传播。
64 条警告均来自旧 t-PatchGNN 的奇数头 nested-tensor 优化提示。

另外从修改前 `f5458d5` 动态载入旧算子，对 `irgraph`、`voronoi`、`voronoi_h`
各自严格加载相同 state_dict，在两层不规则输入上输出均逐元素完全一致。
新增残差投影仅在输入/输出维度不同时有参数；旧模型仍使用恒等残差。

脚本检查覆盖六模型主版本与密度消融、默认 batch size 保留、显式 batch 覆盖、
seed 3407、显式 `cuda:1`、含空格路径与拒绝覆盖旧结果。
另通过 `all_six` 生成六模型 × 五数据集 × 两版本的 60 份计划配置。
Python 关键名称/语法检查、shell 语法检查和 `git diff --check` 通过。

## 真实数据 CPU 检查

默认网络宽度检查 `results/voronoi_h_4d_default_shape_20261125`：
六模型 × DIST，保留配置宽度/层数，每 split 一个窗口、1 epoch、诊断 batch=1，**6/6 成功**。

完整数据/版本矩阵的诊断输出在 `results/voronoi_h_4d_smoke_20261125`：
六模型 × Activity、MIMIC、PhysioNet、USHCN、DIST × 两版本，诊断小网络、
每 split 一个样本/窗口、1 epoch、batch=1，**60/60 成功**。
逐项核对 60 份导出协议的同维编码、4d 输入、d 维输出、无前置融合 MLP、无独立密度偏置，
以及主版本/消融是否启用密度；30 对模型/数据集的两个版本参数数量全部相等。
合计 66 次真实数据训练、验证、测试与导出成功。
诊断 batch=1 仅用于本地验证，正式脚本保留已有配置的 batch size。
所有诊断结果写入新的忽略目录，未修改或提交数据与历史实验输出。
本地 PyTorch 2.14.1 CPU 检查不代表正式 GPU 准确率实验。
