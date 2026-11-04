# ExEnGraphTransformer：内容注意力与时间/变量图偏置

新模型 `ExEnGraphTransformer` 在观测节点上使用多头注意力和 FFN，替换 ExEnAligned 的
Chebyshev 与低秩图卷积。保留原数据接口、数值/时间编码、预测 token、角色/状态编码、
分阶段的同变量时间传播和 X→Y 传播、全局残差、线性预测头，以及 DAG 评估口径。
不使用 Sandwich、超节点或 CSR 图卷积。新配置和结果使用独立目录。

## Graphormer 与本模型

Graph Transformer 是一类模型，Graphormer 是其中一种具体架构。
Graphormer 在标准 Transformer 中加入节点度数编码、最短路径空间偏置和路径上的边信息编码，
使注意力能够识别图结构，而不只是节点内容。
原论文：[Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234)。

本模型借鉴“内容注意力 + 结构偏置”的思路，把结构信息换成时间、变量关系和内外生方向掩码。
它不是原版 Graphormer 的完整复刻，不计算度数/最短路径，也不把注意力权重解释为已识别的因果关系。
“去掉聚合”在这里指不做超节点压缩，注意力内部的加权消息求和仍然存在。

## 当前公式

初始表示沿用 ExEnAligned：已知观测用训练 MinMax 空间的数值编码加时间编码，
查询用预测 token；再加入角色和历史/已知未来/查询状态编码。标签在编码前移除。
每层具有独立时间编码器，令 $z_i=\operatorname{LN}_{attn}(h_i)$。
对关系 $r$ 和头 $a$：

$$
q_i^a=W_{Q,r}^a z_i+b_{Q,r}^a,\quad
k_j^a=W_{K,r}^a z_j+b_{K,r}^a,\quad
v_j^a=W_{V,r}^a z_j+b_{V,r}^a.
$$

时间阶段保留 X-time、Y-time 两组独立投影，只允许同变量传播：

$$
e_{ij}^{a,r}=\frac{\langle q_i^a,k_j^a\rangle}{\sqrt{d_h}}
+\gamma_{t,r}^a\langle p_i,p_j\rangle.
$$

X→Y 阶段另加变量偏置：

$$
e_{ij}^{a,XY}=\frac{\langle q_i^a,k_j^a\rangle}{\sqrt{d_h}}
+\gamma_t^a\langle p_i,p_j\rangle
+\gamma_v^a\langle U_{v_i}^a,V_{v_j}^a\rangle.
$$

所有关系都要求有效节点、源不是预测查询、$t_j\le t_i$；额外的角色与同变量限制沿用原模型。
不使用数组下三角位置替代物理时间比较，相同时间的合法已知节点可以发送消息。
每层均不允许未来信息回到历史，查询在所有层都不作为 K/V。

$$
\alpha_{ij}^a=\operatorname{masked\_softmax}_j(e_{ij}^a),\quad
o_i=W_{O,r}\operatorname{Concat}_a\left(\sum_j\alpha_{ij}^a v_j^a\right)+b_{O,r},
$$
$$
\tilde h_i=h_i+\operatorname{Dropout}(o_i),\qquad
h_i'=\tilde h_i+\operatorname{Dropout}\left(\operatorname{FFN}(\operatorname{LN}_{ffn}(\tilde h_i))\right).
$$

FFN 为 Linear → GELU → dropout → Linear。跨变量阶段仅更新 Y，X 保持原状态。
空邻域的注意力消息及输出投影偏置均置零，仍执行自身残差与 FFN；无外生模式下所有层都参与训练。
全局残差与线性输出头不变，只对未来有效 Y 查询计算训练和测试误差。

## 参数与消融

当前所有数据集 YAML 配置为 `hid_dim: 32`、`num_heads: 4`、`dropout: 0.3`、
`n_layer: 4`、`n_cross_layer: 3`，共 7 层，每头内容维度 8。
FFN 宽度倍数仍为 2，`attention_dropout: 0.0`；这里的 dropout 0.3 用于残差分支和 FFN。
模型构造函数的默认参数保持原值，运行实验时以 YAML 显式参数为准。
注意力、FFN、角色和变量嵌入会使参数量增加，`metrics.json` 会记录参数数量。
与旧图卷积比较时，这属于算子/结构替换，不是相同参数量的单因素实验。

- `use_time_bias: true`、`use_variable_bias: true`：默认启用图偏置。
- `time_bias_init: 1.0`，`variable_bias_init: 0.01`：可学习的逐头系数。
- 变量偏置使用小初始化，避免随机变量关系在训练初期压过内容项。这与旧版初始化 1 不同；
  如需控制该因素，应额外用 `variable_bias_init: 1.0` 比较，不能把全部变化归因于 Q/K/V。
- `attention_backend: sdpa`；`reference` 提供显式 softmax 对照，不作为性能模式。
- `receiver_chunk_size: 512`：控制单次注意力接收块大小，不删边、不做 top-k。
- `attention_dtype: float32`：默认保证 FP32 对照。
- `auto`：CUDA 上支持时选 BF16，否则 FP16；CPU 保持 FP32。
- `bfloat16` / `float16`：指定 CUDA 注意力精度；CPU 仍使用 FP32。

这些精度设置只作用于注意力内部 Q/K/V 计算；残差、参数和 DAG 指标处理继续使用原来的路径。
`enable_autocast` 仍为 false。切换精度会改变数值舍入，应记录运行配置。

## 效率实现与边界

复用原模型排序后的合法源索引与接收行度数，按同变量或 X→Y 分组运行注意力。
实现没有创建 CSR 稀疏矩阵，也没有逐边 sampled_addmm 或 segment softmax。

为避免单独创建完整的 $N\times N$ 图偏置矩阵，使用增广 Q/K：

$$
q_i'=[q_i/\sqrt{d_h},\gamma_t p_i,\gamma_v U_{v_i}],\qquad
k_j'=[k_j,p_j,V_{v_j}],\qquad v_j'=[v_j,0,0].
$$

调用 SDPA 时设置 `scale=1`，其内积严格对应上述内容项和图偏置；输出裁回原头维度。
时间阶段省略变量项，并将通道补零到 8 的倍数。
每次只生成接收块的时间前缀 mask；训练时使用 checkpoint 避免保留所有块的 mask/注意力中间量。
保持完整合法边集合，计算量仍随合法节点对数量增长。

SDPA 会根据设备、精度、形状与掩码选择内核，不保证每次使用 FlashAttention。
FP32、额外通道填充和小批量都可能影响收益；DIST 的大节点数仍需实际评估。
[PyTorch SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
说明了不同实现的选择及限制。

## 运行两组实验

使用原 `data/processed/` 数据，配置位于 `config/ExEnGraphTransformer/processed/`。
两组默认都跑 12 个非 DIST 数据集，沿用同一划分、窗口、目标 mask、零值缺失规则和训练缩放统计。

```bash
conda activate strgcn-exen
# 有历史 X、已知未来 X；默认 FP32 注意力
DEVICE=cuda:0 bash scripts/run_exen_graph_transformer.sh
# 同一个模型，无历史/未来 X
DEVICE=cuda:0 bash scripts/run_exen_graph_transformer_no_exogenous.sh
```

GPU 自动选择注意力精度时，在相应命令前增加 `ATTENTION_DTYPE=auto`：

```bash
ATTENTION_DTYPE=auto DEVICE=cuda:0 bash scripts/run_exen_graph_transformer.sh
ATTENTION_DTYPE=auto DEVICE=cuda:0 bash scripts/run_exen_graph_transformer_no_exogenous.sh
```

CPU 使用 `DEVICE=cpu`。`RUN_TAG` 可为每组指定自己的输出名，`PYTHON` 可指定解释器。
默认输出：

```text
results/exen_graph_transformer_known_future/<数据集>/
results/exen_graph_transformer_no_exogenous/<数据集>/
model_states/exen_graph_transformer_known_future/<数据集>/
model_states/exen_graph_transformer_no_exogenous/<数据集>/
```

单独运行 NP：

```bash
python -u main.py --config_path config/ExEnGraphTransformer/processed/NP.yaml --device cuda:0 \
  --model_save_dir_path model_states/exen_graph_transformer_known_future/NP \
  --result_save_dir_path results/exen_graph_transformer_known_future/NP
```

无外生单次命令增加 `--no-exogenous`，并更换两个输出路径；自动精度增加 `--attention-dtype auto`。
需要 DIST 时增加 `INCLUDE_DIST=1`。新 DIST.yaml 仍沿用仓库原窗口；若采用 15 分钟→3 分钟任务，
应先把其 `history_len` 改为 `900`、`forecast_len` 改为 `180`（秒）。
新模型必须重新训练，不加载旧 GCN checkpoint。主要比较 test `mse_norm`、`mae_norm`。

## 核验与实际 GPU 测速

```bash
python -m pytest -q tests
# 同一实际 NP 批次，比较 ExEnAligned 与 Graph Transformer，测前向+反向
python examples/benchmark_exen_graph_transformer.py --dataset NP --device cuda:0 \
  --attention-dtype auto --batch-size 4 --warmup 5 --steps 10 \
  --output reports/graph_transformer_gpu_benchmark.json
```

测速不更新参数，不代表准确率或完整 epoch 吞吐；记录设备、参数量、计时和 CUDA 峰值已分配显存。
可添加 `--no-exogenous` 比较无外生场景。NP 的本地 CPU 短批次记录见
`reports/graph_transformer_cpu_benchmark.json`。GPU 性能和完整训练效果需要在你的设备上验证。

本次 149 项测试通过；12 个实际非 DIST 数据集的两种输入模式共完成 24 次首训练窗口前向/反向，
并核对每个划分首窗口的目标样本和窗口列表一致，记录在 `reports/graph_transformer_smoke.json`。
测试覆盖 SDPA 与显式公式的输出/所有参数梯度一致、内容相关注意力、目标及晚时刻输入无泄漏、
空邻域、padding、节点乱序、无外生时三层 FFN 可训练，以及两种模式的完整 CLI 和 DAG 量纲。
本地 NP CPU、batch=4、FP32、3 次计时的前向/反向中位数约 0.0676 秒（ExEnAligned）和 0.0197 秒（新版），
约 3.44 倍；当时使用接收块 128/512，新版参数量 9294、旧版 2337。
该历史基准使用新版 2+1 层、隐藏维度 16、2 个头、dropout 0，
不代表当前 4+3 层、隐藏维度 32 配置的速度。
这是当前实现和配置的短批次结果，不是同参数量或同分块大小的纯算子对照，CUDA 尚未实测。
