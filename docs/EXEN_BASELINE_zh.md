# Ex-En baseline：实现与运行说明

本实现基于原仓库 `31194f9`，新增 `ExEn`，原 STRGCN 模型仍保留。
模型使用统一观测图、角色/状态编码和三类有向关系，移除 Sandwich 与超节点聚合。

你提供的 `time/values/mask/names` 格式 processed 数据，请使用 [processed 专用入口与 conda 说明](EXEN_PROCESSED_zh.md)。该入口按最新要求将所有原始 0 视为缺失。下文的预切分长表入口继续保留。

## 1. 快速运行

在仓库根目录执行（Python 3.12）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-exen.txt
python -m pytest -q tests/test_exen.py
python examples/make_exen_demo.py
python main.py \
  --config_path config/ExEn/ExEn_npz.yaml \
  --model_save_dir_path model_states/exen_demo \
  --result_save_dir_path results/exen_demo
```

默认 CPU。使用 GPU 时将配置中的 `train_config.device` 改为 `cuda:0`。
初版验证了 CPU；未测试 CUDA、混合精度和分布式运行。
最小依赖不含 GP/Sandwich；运行原始聚合模型仍需其原有依赖，另需安装 `gpytorch`（原 requirements.txt 未列出）。

输出沿用原训练器：模型目录内有配置、checkpoint 和最佳模型；结果目录内有
`result.json`、`test_y_pred.npy`、`test_y_true.npy`。预测数组包含所有节点，**只有查询掩码位置属于评估目标**；掩码可以由同一数据集的 collate 输出获得。
默认报告训练集缩放后的误差，`do_inverse_scaling: false`。原 LongMinMaxScaler 没有训练器所需的 `inverse_transform_mx`，请勿直接开启该选项。

## 2. 使用真实数据

提供 NPZ 文件，包含 `train_data`、`valid_data`、`test_data` 三个数组，每行为：

```
[ts_id, var_id, timestamp, value]
```

训练/评估数组为有限值 `[N,4]`，ID 为整数。每个序列内时间可以不规则、各变量无需对齐；缺失观测不提供行。
先按研究协议切分数据，避免滑窗跨越训练和评估边界。NPZ 加载器不重新切分数据。

修改配置：

- `global_config.exen_data_path`：NPZ 路径。
- `history_len`、`forecast_len`、`time_end`：与数据时间戳单位一致。沿用原窗口规则，历史为 `[start,start+history_len)`，未来为 `[start+history_len,start+history_len+forecast_len)`。
- `use_time_chunk: true` 时开启原滑窗，可配置 `chunk_stride`。
- `exogenous_var_ids`：外生变量的**原始 ID**；其余训练变量视为内生。
- `known_future_exogenous_var_ids`：发布预测时确实已知未来值的外生变量 ID，必须为外生集合的子集。默认空列表，表示不提供未来外生值。

例如 demo 中外生是 10、30，内生是 20，但只有 10 的未来值已知。
变量 30 的未来行会变为无效节点，不参与关系、编码或损失。
不要把事后测到的外生值配置为“已知未来”。

训练集建立原始 ID 到嵌入索引的映射，验证和测试复用；评估时允许某些变量缺失，但不接受训练时从未出现的变量。
模型不必再手填 `exogenous_var_indices`，数据集会通过 `global_args` 提供。
采用固定 `time_end` 或相对窗口时间归一化，避免利用评估序列的后续最大时间决定缩放。
数据缓存默认禁用，因为上游缓存名没有完整标识输入数据；窗口切分、时间归一化和长表拼接复用原实现。
无未来内生标签的训练窗口会被过滤；全部过滤时明确报错。

也可将现有 Forecasting YAML 中的 `dataset_name` 改为 `ExEnPredictionDataset`，添加上述两组角色 ID；
将 `model_name` 改为 `ExEn`，参数参考示例。需要同时关闭 `enable_agg_mask`、`use_info_agg`、`use_efficient_gcn`、`use_revin`，
改用 `gcn_type: gcn` 和 `gcn_depth: 1`。领域数据的内外生划分需要由研究任务明确指定，不自动推断。

## 3. 结构与源码对应

| 部分 | 处理 |
|---|---|
| 观测级节点 | 保留；不要求规则网格 |
| 数值编码 | 保留 `Linear(1,D)` |
| 连续时间编码 | 直接复用原 `TimeEmbedding`，支持 rbf/rope/transformer |
| 未知内生查询 | 保留 `pred_token`，编码前移除其值 |
| 角色和状态 | 新增外生/内生角色及历史/已知未来/查询状态嵌入 |
| 时间关系 | 保留源码中的时间嵌入内积和可学习尺度 |
| 跨变量关系 | 复用源码中 receiver U / sender V 嵌入内积和可学习尺度 |
| 消息卷积 | 直接调用原 `low_rank_graph_conv`，配合自身变换、ReLU、dropout、LayerNorm 和残差 |
| 预测头 | 保留原 `Linear(D,1)`（单层 MLP），没有额外增加隐藏预测头 |
| Sandwich/GP/超节点 | ExEn 中均不使用 |

实际源码的 EfficientSTRGCNLayer 使用**时间与变量 logits 相加后 softmax**，与前面概念稿中的双线性时间矩阵、乘法关系与带符号除法归一化不同。
此实现以实际源码为准，不声称逐式实现概念稿中的这些公式。
时间阶段仅使用时间 logits；跨变量阶段加入变量 logits（采用源码 bottom 层的 `sqrt(node_dim)` 缩放）。
直接调用低秩卷积时，源节点变换为：

```
u_j = <h_j, Linear(p_j)> * Embedding(var_j)
m_i = sum_j softmax_j(masked_logits_ij) * u_j
```

在此基础上加入节点自身变换与残差。没有另加因果损失或滞后模块。
`n_layer` 控制同变量时间阶段深度，`n_cross_layer` 控制其后的外生→内生阶段深度。
历史和未来共享同一套关系参数，不按状态建立两套网络。
跨变量阶段固定外生表示；无跨变量入边的节点保留其上一阶段表示。
历史内生节点在跨变量阶段更新后，不再向查询传播；未来查询通过同变量历史表示及外生入边获取信息。
初版仅使用未来查询 MSE，不额外监督历史重建。

## 4. 掩码与目标泄漏防护

邻接矩阵维度为 `[batch, receiver, sender]`，三个候选关系为：

1. 外生同变量时间边：X_j→X_i，变量相同且 `t_j <= t_i`。
2. 内生同变量时间边：Y_j→Y_i，变量相同且 `t_j <= t_i`，且 j 必须为历史观测。
3. 外生→内生边：X_j→Y_i，且 `t_j <= t_i`。

所有关系均剔除无效/填充节点。未来内生查询从不作为图消息源，也不建立不同内生变量之间、不同外生变量之间或内生→外生的边。
允许同时间戳的外生→内生关系，与讨论中的 `<=` 定义一致。
每一层都在 softmax **之前**应用掩码；空邻域权重为零。
自身残差仅是同一节点的状态变换，不是查询向其他节点发消息。

查询值和填充值在数值编码前通过 `torch.where` 替换；即便查询占位值为 NaN/Inf，也不会进入 encoder 或其梯度。
不使用批次/序列共享的值归一化，避免较晚的未来外生值经归一化统计影响较早查询。
NPZ 入口的缩放器仅拟合训练集，修正了原入口使用 train+valid 的行为（仅对 ExEn 启用）。
损失先选择查询节点再做差，评估掩码同时包含有效节点掩码。

这些约束限制信息流，不能据此把学习到的边权解释为已识别的因果效应。
与原 STRGCN 对比时应统一预测的内生变量、可用未来外生信息、数据切分、缩放与评估掩码。

## 5. 无标签推理接口

推理不需要未来内生真实数值，也无需依赖训练数据集中“出现标签的行”来生成查询。直接构造任意查询时间：

```python
import torch
from models import ExEn

model = ExEn(num_vars=2, exogenous_var_indices=[0], hid_dim=16).eval()
# 0=历史外生，1=历史内生，2=已知未来外生，3=未来内生查询
batch = dict(
    batch_value=torch.tensor([[1., 2., 1.5, float('nan')]]),
    batch_timestamp=torch.tensor([[0., 0., 1., 1.]]),
    batch_var_idx=torch.tensor([[0, 1, 0, 1]]),
    batch_node_type=torch.tensor([[0, 1, 2, 3]]),
    batch_pad_mask=torch.ones(1, 4),
    batch_pred_mask=torch.tensor([[0, 0, 0, 1]]),
)
with torch.no_grad():
    all_nodes = model(**batch)  # [1,4,1]
    prediction = all_nodes.squeeze(-1)[batch['batch_pred_mask'].bool()]
```

生产推理需加载训练配置与权重，并复用训练变量映射和缩放器；所有有效历史时间必须严格早于所有未来时间。
可省略未来外生节点；可提供不同内生变量、不同时间的多个查询。
`batch_node_type` 必须显式提供，禁止根据目标数值推断节点身份。

## 6. 验证与计算量

19 项测试通过，覆盖关系定义、目标替换不变性、查询输入零梯度、已知输入非零梯度、三种时间编码、单样本/多样本、
较晚外生值不影响较早查询（多层）、不同内生变量无消息路径、非有限填充值、空邻域、角色配置、跨 split 映射、
缺省未来外生、批次隔离、查询置换、训练/评估、权重保存加载，以及完整 CLI 与训练集专用缩放统计。

CPU 合成实验（seed 3407，24/8/8 个训练/验证/测试序列，10 epoch）已通过原 `main.py` 完整运行。
环境：Python 3.12，PyTorch 2.15.0，NumPy 2.5.4，pandas 3.1.0。
本次测试集缩放空间 MSE 为 0.00564923，MAE 为 0.0628842；仅用于证明流程可训练，不代表真实数据性能。
尚未运行 USHCN、Activity、PhysioNet 或 MIMIC 真实数据实验，因为没有提供数据。

默认整图模式构建稠密 `[B,L,L]` 掩码与关系张量；processed 配置新增 `receiver_chunk_size` 分块及训练重算，减少中间张量内存，关系计算总量仍为平方级。
删除 Sandwich 后没有继承原超节点加速；长序列应先减小 batch/窗口长度，后续再评估稀疏边实现。
不支持原 Chebyshev 多阶模式；通过堆叠直接受掩码控制的图层增加深度，以便明确约束每一步消息方向。
