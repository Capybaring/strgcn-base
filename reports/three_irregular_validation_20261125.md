# GraFITi / CRU / LatentODE 密度融合验证（2026-11-25）

在原 master 接入三种 backbone，默认沿用已有 `voronoi_h`。
保留三者的原始历史编码算子和预测头，来源固定到作者仓库提交，见
[来源与适配边界](../baseline/irregular_backbones/README.md)。
原有数据、旧方法、历史结果及用户原有四个未跟踪文件未修改、未提交。

## 自动验证

以下联合回归 **317 项全部通过**，其中新适配器文件包含 60 项：

```text
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

本地使用 PyTorch 2.14.1、CPU、`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
联合回归耗时 81.76 秒；58 条提示均来自旧 t-PatchGNN 的奇数头 nested-tensor 优化提示。

覆盖内容：

- 三模型 × 五数据集 × base、零密度匹配消融、voronoi_h 的前向/反向与有限梯度。
- 原头输出等价；GraFITi 暴露表示后接原头与默认原 forward 逐元素一致。
- CRU 隐状态、正方差和完整均值解码器与直接调用原滤波/解码算子逐元素一致，方差头获得梯度。
- LatentODE 原 ODE-RNN 后验参数获得梯度，KL 非负；常导数 ODE 的解析例子验证
  从历史初始时刻积分，而非错误地从未来首个查询时刻开始。
- 未来 Y 改为 NaN 不改变预测；对未来 Y 输入的梯度为零，未来 X 可以影响预测。
  CRU 信息流梯度测试对解码器偏置作受控设置，避免随机初始化的 ReLU 死区混淆路径判断。
- 不同长度样本联合推断保持单样本结果；无 X 时融合为恒等映射。
- 分批脚本覆盖模型/数据矩阵、seed 3407、保留配置 batch size、显式 batch 覆盖、
  `--device cuda:1`、含空格路径以及拒绝覆盖旧输出目录。

关键 Python 名称/语法检查、两个 shell 入口的 `bash -n` 和 `git diff --check` 通过。

## 真实数据 CPU 短程训练

全部使用新目录，不覆盖或提交诊断模型与历史输出：

| 检查 | 模型及数据 | 结果 |
| --- | --- | --- |
| `results/three_irregular_smoke_20261125` | 三模型 × Activity、MIMIC、PhysioNet、USHCN、DIST | 15/15 成功 |
| `results/three_irregular_default_shape_20261125` | 三模型 × MIMIC、DIST，保留配置网络宽度和层数 | 6/6 成功 |

两组均为每 split 一个样本/窗口、1 epoch、诊断 batch=1；第一组使用小网络。
该 batch=1 只用于本地诊断，运行脚本和正式配置仍保留各数据集默认 batch size。
训练、验证、测试、检查点和协议导出均完成。导出协议确认
`density_location=node`、`density_node_encoding=true`、
`density_attention_encoding=false`、`density_bias=none`。
短程指标不用于评价准确率；未在本地进行正式 GPU 实验。

## 实验口径

GraFITi 使用变量宏平均预测 MSE；完整 CRU 使用变量宏平均 Gaussian NLL，
保留 Van Loan 协方差传播、float64 滤波状态和原高斯解码器，插件只融合均值。
LatentODE 保留原 ODE-RNN、生成 ODE 和原线性头，训练用后验采样与预测 MSE+KL，
评估取后验均值；不声称复现作者的重要性加权似然训练。
三者均按统一验证 MSE 早停，正式比较应使用同 backbone 的三个版本。
完整命令、默认 batch size 及依赖见
[运行说明](../docs/IRREGULAR_BACKBONE_FUSION_zh.md)。
