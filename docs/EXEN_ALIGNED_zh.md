# ExEnAligned：保留内外生设计，统一 STRGCN 算子

本次修改只对齐与内外生设计无关的四项计算方式。新模型名为 `ExEnAligned`，
原 `ExEn`、已有配置及结果保留用于复现；重新实验请使用 `config/ExEnAligned/processed/`。

## 保留与统一的内容

内外生设计保持原结构：

- 保留内外生角色嵌入、历史/已知未来/预测查询状态嵌入。
- 先 2 层同变量时间传播，再 1 层外生→内生传播，总计 3 层。
- 时间阶段仍分别处理同变量 X→X、同变量 Y→Y，保留两类关系独立的权重。
- 跨变量阶段仍只允许 X→Y，X 在该阶段保持不变。
- 保留原节点检查、预测 token、全局残差和线性预测头。

与 `STRGCNDirect` 统一的算子：

| 项目 | 统一后的计算 |
|---|---|
| 变量打分尺度 | 不除以 $\sqrt{d_v}$；尺度/温度参数均初始化为 1 |
| 卷积 | Chebyshev $T_0,T_1$，默认 `gcn_depth: 2` |
| 低秩分解 | 先图传播，再用接收节点时间/变量权重变换 |
| 时间特征 | 每个传播层拥有独立的时间编码器 |
| 层更新 | ReLU、dropout、门控残差、LayerNorm，与联合版相同 |

两版共享同一个 `receiver_cheb_convolution` 函数，避免再次产生两套卷积定义。
原 ExEn 的独立 `self_linear` 被 $T_0$ 对应的自身项替代。
初始节点编码仍有一组时间编码器，此外每个传播层独立拥有时间编码器。

**无外生时仍执行全部三层。**最后的跨变量层没有外生邻居时，$A=0$，但保留 $T_0H=H$
对应的自身变换、残差和归一化；不沿用旧 ExEn 的“没有 X 就跳过整层”规则。
这也是将自身变换统一为 Chebyshev $T_0$ 后的边界行为。
跨变量层的 X 节点依旧不更新。

## 公式

时间阶段中每种关系 $r\in\{X\text{-time},Y\text{-time}\}$ 的邻接：

$$
A^r_{ij}=\operatorname{masked\_softmax}_j
\left(\tau^r_t s^r_t\langle p_i^{(\ell)},p_j^{(\ell)}\rangle,M^r_{ij}\right).
$$

仍只用同变量时间关系打分。同变量邻域中的变量嵌入内积对每行是常数，
若加入联合版变量项，也会在该邻域的 softmax 中抵消。

跨变量阶段为：

$$
A^{XY}_{ij}=\operatorname{masked\_softmax}_j
\left(\tau_t s_t\langle p_i^{(\ell)},p_j^{(\ell)}\rangle+
\tau_v s_v\langle U_{v_i},V_{v_j}\rangle,M^{XY}_{ij}\right).
$$

各类掩码都要求有效节点、$t_j\le t_i$，且源节点不是预测查询。
时间关系额外要求同变量及对应 X/Y 角色；跨变量关系要求源 X、接收 Y。

每个关系采用与联合版相同的计算：

$$
Z_0^r=H,\qquad Z_1^r=A^rH,\qquad
g_i^r=\sum_{k=0}^{1}\langle Z_{k,i}^r,W_T^r p_i^{(\ell)}+b_T^r\rangle w_{S,k}^r(v_i).
$$

每个接收节点只保留所属关系域的输出：时间层 X 节点选 X-time 分支，Y 节点选 Y-time 分支；
跨变量层仅更新 Y。$T_0$ 不在 X、Y 两个时间分支之间重复累加。
随后加层偏置并使用联合版相同的门控残差与 LayerNorm。

本次仍保留角色编码、硬关系限制和分阶段传播三项设计，
因此联合版与新版 ExEn 的比较是“整套内外生设计”的对照，并非仅角色嵌入的单因素实验。

## 有外生、无外生两组运行

数据仍位于 `data/processed/`。全部配置与联合版使用相同的数据划分、时间窗口、
训练参数、随机种子、零值缺失规则和 DAG 标准化评估空间。
默认已知未来输入为 `all`；无外生模式同时删除所有历史和未来 X 节点，仅保留历史 Y 与未来 Y 查询。
窗口、目标值与目标 mask、变量 ID 映射和训练缩放统计保持一致。

```bash
conda activate strgcn-exen
# 有历史 X、已知未来 X，12 个非 DIST 数据集
DEVICE=cuda:0 bash scripts/run_exen_aligned.sh
# 同一个 ExEnAligned，无历史/未来 X，12 个非 DIST 数据集
DEVICE=cuda:0 bash scripts/run_exen_aligned_no_exogenous.sh
```

CPU 改用 `DEVICE=cpu`。可用 `PYTHON` 指定解释器、`RUN_TAG` 指定各组自己的实验目录。
输出目录为：

```text
results/exen_aligned_known_future/<数据集>/
results/exen_aligned_no_exogenous/<数据集>/
model_states/exen_aligned_known_future/<数据集>/
model_states/exen_aligned_no_exogenous/<数据集>/
```

单独运行 BE：

```bash
python -u main.py --config_path config/ExEnAligned/processed/BE.yaml --device cuda:0 \
  --model_save_dir_path model_states/exen_aligned_known_future/BE \
  --result_save_dir_path results/exen_aligned_known_future/BE
python -u main.py --config_path config/ExEnAligned/processed/BE.yaml --device cuda:0 --no-exogenous \
  --model_save_dir_path model_states/exen_aligned_no_exogenous/BE \
  --result_save_dir_path results/exen_aligned_no_exogenous/BE
```

需要 DIST 时可加 `INCLUDE_DIST=1`；其配置沿用仓库原窗口。使用 15 分钟→3 分钟任务时，
先把新 `config/ExEnAligned/processed/DIST.yaml` 中 `history_len` 设为 `900`、`forecast_len` 设为 `180`（秒）。
这组实验应重新训练，不载入旧 ExEn 或联合版的 checkpoint。

## 核对实验及验证

在 `metrics.json` 的 protocol 中核对：`model_name: ExEnAligned`、各组的 `use_exogenous`、
`known_future_exogenous_var_ids`、`graph_operator`（统一算子设置）。比较测试集 `mse_norm`、`mae_norm`。

```bash
python -m pytest -q tests
```

测试覆盖原 STRGCN Chebyshev 算子的数值对照、dense/CSR 输出及全部参数梯度、
多层查询标签与晚时刻输入不泄漏、X 不接收 Y、无外生时三层均参与训练、
两种输入模式的完整训练/评估 CLI、DAG 量纲及批量入口。

本次共 126 项测试通过。12 个实际数据集的两种输入模式共完成 24 次首训练窗口的 CPU 前向/反向检查，
每个模式三层均有训练梯度，预测查询输入梯度为零；有外生模式的已知未来输入梯度非零。
每个划分首窗口的目标样本与两组完整窗口列表也已核对一致。
记录见 `reports/exen_aligned_smoke.json`；这些短窗口检查不代表完整训练精度，CUDA 尚未实测。
