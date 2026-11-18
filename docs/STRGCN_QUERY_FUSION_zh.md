# STRGCN 外生编码＋现有查询交叉注意力

新方法名为 `strgcn`，不覆盖旧 `v1`、配置、数据和已有结果。

## 方法

仅替换 X 的编码路径，backbone 的 Y 历史编码器和预测头不变。
复用仓库 `DirectSTRGCNLayer`，与原 `EfficientSTRGCNLayer` 的 Middle
阶段保持相同的关系打分、Chebyshev 递推和时间/变量低秩权重变换。
这是对原算子的复用，不是复制整套 Bottom–Middle–Top Sandwich 或隐节点机制。

- 入口：原 STRGCN 的 Linear(1,D) + RBF 时间编码；不添加旧插件的变量向量。
- 图关系：softmax(time_temp*time_scale*P P^T + val_temp*val_scale*U V^T)。
  保留 Middle 原有缩放；不额外除以 sqrt(node_dim)，不增加内容 QK 打分。
- 特征变换：复用原低秩 Chebyshev 算子；先算 T_k(A)H，再在接收方使用
  时间权重与第 k 阶变量权重。它与一阶 GCN 的“先来源变换再传播”次序不同。
- 默认一层图编码、Chebyshev 两个阶项、变量关系秩 16；保留原激活、层归一化、
  层内残差和编码器外层残差。参数可用 x_layers/x_depth/x_node_dim 修改。
- 输出仍对应原始 X 观测位置；将上下文化 H_X 接入现有 DynamicLagFusion 的 K/V。
  Q、跨 X→Y 的时间/变量关系偏置、交叉注意力和 G 残差沿用当前 V1。
- X 内部关系参数和跨 X→Y 关系参数独立训练；无历史 Y→X 关系迁移。

## 全窗口 X 可见协议

`strgcn` 和 `v1_all` 都允许输入窗口内全部有效、已提供的历史与未来 X
影响任意 Y 查询，包括时间晚于查询的 X。X 内部同样无时间方向或局部窗口限制。
只保留有效观测掩码；未来 Y 标签和 Y 查询永不成为消息来源。
NoExo 下插件恒等，HistExo 下只接收历史 X，均保持原数据角色约束。
“未来 X”指数据集实际提供的外生观测，不补造未观测值，不读窗口之外的数据。

旧 v1/cross/v2 默认仍使用 source_time <= query_time，以便保留已有对照语义。
`v1_all` 是只放开 X 时间限制、仍逐点编码的对照；与 `strgcn` 比较才能分离
上下文编码带来的收益。base 是 Y-only；MLP 保留原按查询取最近可见 X 的输入方式，
其表示/可见性不与全窗口注意力等同，不应仅凭与 MLP 的差异归因于编码器。
结果 protocol.fusion 会记录实际 X 编码器、可见性和掩码。

这改变了实验条件：全窗口未来 X 是已知条件/oracle 设置，与旧因果可见性结果分开报告。
本次不扩展 V2 的历史动态滞后迁移；STRGCN 编码与 V2 组合会明确报错。

## 运行

在远程仓库中更新代码后，直接执行：

```bash
bash scripts/run_strgcn_query_fusion.sh
```

默认 Hi-Patch、四个数据集、base/mlp/v1_all/strgcn，单 seed 3407，逐个运行共 16 次。
默认 GPU 0，最多 30 epoch，保留现有早停。每次新建结果目录，拒绝覆盖已有目录。

```bash
bash scripts/run_strgcn_query_fusion.sh tpatchgnn
bash scripts/run_strgcn_query_fusion.sh mtand
```

只运行一个模型、一个数据集、新方法：

```bash
bash scripts/run_query_fusion.sh hipatch activity strgcn
```

环境变量与原脚本相同：GPU、PYTHON、TPATCH_DATA_ROOT、RUN_DIR、DEVICE、DRY_RUN。
训练仍为从头端到端训练；不加载额外预训练权重。
