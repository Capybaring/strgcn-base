# STRGCNDirect：去掉 Sandwich 与超节点聚合的对照实验

`STRGCNDirect` 是一个独立模型。使用原 STRGCN 的数值编码、时间编码、
时间与变量关系联合打分、低秩 Chebyshev 图卷积、门控残差与线性预测头。
删除观测→超节点→观测的 Sandwich、Gaussian/GP 聚合和超节点；
每层直接在原始观测节点与预测查询节点上运算。这里删除的“聚合”指超节点聚合，
图卷积内部的加权消息求和仍然存在。

图计算与节点嵌入不使用内生/外生角色，也没有 ExEn 的三类关系掩码或
先时间、再外生→内生的分阶段传播。变量 ID 仍用于原 STRGCN 的可学习变量关系和卷积权重。
为确定哪些未来值可作为输入、哪些值需要预测，数据集仍保留输入列/目标列的划分；
这不限制模型内的变量传播方向。默认把所有非目标列的已知未来观测加入输入。

## 公式与实现范围

记节点为 $(v_i,t_i,x_i)$，$q_i=1$ 表示待预测查询，$m_i=1$ 表示有效节点。
沿用原 STRGCN 的初始化：已知节点为 $h_i^{(0)}=W_xx_i+b_x+p(t_i)$，
查询节点为可学习 `pred_token`。查询标签在进入数值编码器之前被替换为零。

每层独立计算时间特征 $p_i^{(\ell)}$ 和变量嵌入 $U_{v_i}^{(\ell)},V_{v_j}^{(\ell)}$：

$$
e_{ij}^{(\ell)}=
\tau_t^{(\ell)}s_t^{(\ell)}\langle p_i^{(\ell)},p_j^{(\ell)}\rangle+
\tau_v^{(\ell)}s_v^{(\ell)}\langle U_{v_i}^{(\ell)},V_{v_j}^{(\ell)}\rangle.
$$

这是仓库 `EfficientSTRGCNLayer` 中间层的联合关系公式，直接应用到观测节点。
保留中间层的尺度定义，变量项不额外除以 $\sqrt{d_v}$。
为延续预测任务的防泄漏约束，使用：

$$
M_{ij}=m_im_j(1-q_j)\mathbf1[t_j\le t_i],\qquad
A_{ij}^{(\ell)}=\operatorname{masked\_softmax}_j(e_{ij}^{(\ell)},M_{ij}).
$$

任意变量均能向任意变量发送消息，源时间不得晚于接收时间。查询在所有层、所有阶都不作为消息源。
未来已知观测可以影响同一时刻及之后的查询；更晚的已知观测不能经多层间接影响更早查询。
这与原 Sandwich 中间层不受限的 softmax 有区别，是该预测对照保留的时间与查询掩码。
这里的时间约束不等于识别结构因果关系。

默认使用原 `low_rank_cheb_graph_conv` 的传播后分解权重形式：

$$
Z_0=H^{(\ell)},\quad Z_1=A^{(\ell)}H^{(\ell)},\quad
Z_k=2A^{(\ell)}Z_{k-1}-Z_{k-2},
$$
$$
w_{T,i}^{(\ell)}=W_T^{(\ell)}p_i^{(\ell)}+b_T^{(\ell)},\qquad
g_i^{(\ell)}=\sum_{k=0}^{K-1}\langle Z_{k,i},w_{T,i}^{(\ell)}\rangle
w_{S,k}^{(\ell)}(v_i)+b^{(\ell)},
$$
$$
h_i^{(\ell+1)}=\operatorname{LN}\left[
\lambda_\ell\operatorname{Dropout}(\operatorname{ReLU}(g_i^{(\ell)}))+
(1-\lambda_\ell)h_i^{(\ell)}\right],
$$
$$
\hat y_i=W_o\left[\lambda_g h_i^{(L)}+(1-\lambda_g)h_i^{(0)}\right]+b_o.
$$

默认 `n_layer: 3`、`gcn_type: cheb`、`gcn_depth: 2`，即每层使用 $T_0,T_1$。
支持更高阶；`gcn_type: gcn, gcn_depth: 1` 则采用原 `low_rank_graph_conv` 的先源节点变换、再传播形式。
这一实现针对原有效 Sandwich 路径的中间关系与卷积，不使用原未完整接入模型 forward 的非高效分支。

## 数据、评估与配置

数据仍放在仓库 `data/processed/`，不用移动或重新处理。
13 个独立配置位于 `config/STRGCNDirect/processed/`，沿用 ExEn 对应配置的时间窗口、
划分、零值缺失处理、训练参数与评估协议，不覆盖 ExEn 配置及已有结果。

- 所有原始零值与非有限值视为缺失，并与文件 mask 合并；标准化后合法的零值保留。
- 历史所有变量、已知未来输入变量、未来目标查询均进入图；未来目标真值只用于监督和评估。
- 默认 `known_future_exogenous_var_ids: all`，该字段只决定数据输入可用性。
- 输入 MinMax 与 DAG 评估 StandardScaler 均仅在训练划分拟合。
- 主要比较 `mse_norm`、`mae_norm`，保持之前的 DAG 标准化空间和按窗口平均方式；无 `_norm` 为原始量纲。
- 结果 `metrics.json` 的 protocol 记录 `model_name: STRGCNDirect` 及关系范围。

默认使用 `sparse_csr`，分块计算全部合法边，不做 top-k、不删除合法关系。
同时提供 `relation_backend: dense` 供数值对照。任意变量间连边比 ExEn 更密，
分块存储没有消除边数量的平方增长，大型 DIST 仍可能很慢。
DIST 配置沿用仓库当前时间窗口；如使用你选择的 **15 分钟→3 分钟** 任务，
运行前在新 `DIST.yaml` 的 `data_processor_params` 中设 `history_len: 900`、`forecast_len: 180`（秒）。

## 运行

从仓库根目录执行，使用现有 `strgcn-exen` 环境和数据：

```bash
conda activate strgcn-exen
# 12 个数据集，默认不包含 DIST；显式选择 GPU
DEVICE=cuda:0 bash scripts/run_strgcn_direct.sh
# CPU
DEVICE=cpu bash scripts/run_strgcn_direct.sh
# 需要包含 DIST 时，先核对新 DIST.yaml 的时间窗口
INCLUDE_DIST=1 DEVICE=cuda:0 bash scripts/run_strgcn_direct.sh
```

脚本顺序运行，某个任务失败即停止。输出放在
`results/strgcn_direct_known_future/<数据集>/` 与
`model_states/strgcn_direct_known_future/<数据集>/`。
可通过 `RUN_TAG` 另设实验目录，`PYTHON` 指定解释器。
配置默认 CPU，脚本未设置 `DEVICE` 时使用配置值。

单独运行 BE：

```bash
python -u main.py --config_path config/STRGCNDirect/processed/BE.yaml \
  --device cuda:0 \
  --model_save_dir_path model_states/strgcn_direct_known_future/BE \
  --result_save_dir_path results/strgcn_direct_known_future/BE
```

单独运行其他数据集时替换上述三个路径中的 `BE`。不要复用 ExEn 的 checkpoint；新模型结构不同。

### 同一个联合模型的无外生对照

保留 `STRGCNDirect` 的 3 层、每层 $T_0,T_1$、时间与变量联合打分、权重参数形状和
变量 ID 映射，仅移除所有历史与未来外生观测节点。保留历史目标变量及未来目标查询。
训练/验证/测试窗口、目标真值、目标 mask、训练划分拟合的缩放统计和 DAG 评估空间均保持一致。
这是输入消融，不改回旧 ExEn 算子或分阶段传播。

```bash
# 默认 12 个数据集，排除 DIST
DEVICE=cuda:0 bash scripts/run_strgcn_direct_no_exogenous.sh
# 等价的通用入口
NO_EXOGENOUS=1 DEVICE=cuda:0 bash scripts/run_strgcn_direct.sh
# CPU
DEVICE=cpu bash scripts/run_strgcn_direct_no_exogenous.sh
```

新输出分别位于 `results/strgcn_direct_no_exogenous/<数据集>/` 和
`model_states/strgcn_direct_no_exogenous/<数据集>/`。
与 `strgcn_direct_known_future` 对照，`metrics.json` 的 protocol 应记录
`model_name: STRGCNDirect`、`use_exogenous: false`、`forecast_inputs: Y_history_only`、
`known_future_exogenous_var_ids: []`。
单独运行 BE：

```bash
python -u main.py --config_path config/STRGCNDirect/processed/BE.yaml \
  --no-exogenous --device cuda:0 \
  --model_save_dir_path model_states/strgcn_direct_no_exogenous/BE \
  --result_save_dir_path results/strgcn_direct_no_exogenous/BE
```

`INCLUDE_DIST=1` 和自定义 `RUN_TAG` 在无外生入口同样有效；DIST 时间窗口仍需按前文核对。
训练应从头开始，不复用已有有外生实验的 checkpoint。

## 验证

```bash
python -m pytest -q tests
```

新增测试直接对照仓库原 `low_rank_cheb_graph_conv`（1–4 阶）和 `low_rank_graph_conv`；
覆盖 CSR/dense 输出及所有参数梯度一致、角色置换不变、未来输入有效、查询标签与晚时刻输入不泄漏、
padding、节点乱序、完整训练 CLI 和 DAG 评估空间。
本地验证使用 CPU；不以短窗口冒烟结果作为完整训练效果结论。

联合版首轮验证：89 项测试通过，12 个非 DIST 实际数据集首训练窗口均通过前向/反向、
未来输入非零梯度与查询输入零梯度检查；BE 另完成每个划分最多 4 个窗口、1 epoch 的训练、
checkpoint 载入与测试评估。记录见 `reports/strgcn_direct_smoke.json`。
CUDA 路径尚未在本地实测。

加入无外生批量入口后，93 项测试通过。12 个实际数据集分别核对每个划分首窗口：
无外生样本严格等于有外生样本保留目标变量后的子集，所有窗口列表和变量 ID 映射保持一致；
首训练窗口均通过 3 层 Chebyshev 模型的 CPU 前向/反向与查询输入零梯度检查。
记录见 `reports/strgcn_direct_no_exogenous_smoke.json`，不代表完整训练精度结果。
