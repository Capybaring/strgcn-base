# 同维密度编码与直接拼接（voronoi_h_4d）

本版本把标量密度编码为与值、变量、时间相同的维度，四项直接拼接后进入
Graph Transformer。取消旧 `voronoi_h` 在拼接后的两层融合 MLP。
旧 `voronoi_h` 和 `voronoi_h_no_density` 保留原实现和名称，方便历史对照。
本版本适用于 t-PatchGNN、Hi-Patch、mTAND、GraFITi、CRU、LatentODE 六个 backbone。

## 1. 数学密度保持不变

按每个样本、每个 X 变量的不同观测时刻计算平均间隔 $\mu$。
内部点的左右间隔为 $\delta_i^L,\delta_i^R$：

$$\rho_i=\frac{2}{\delta_i^L+\delta_i^R},\qquad s_i=\log(\mu\rho_i).$$

首尾使用对称延拓的单边密度 $\rho_i=1/\delta_i$，单独一个时刻令 $s_i=0$。
规则采样包括首尾均有 $s_i=0$。重复时间戳共享几何量，但保留所有观测 token。
密度仍由公式计算，没有使用网络估计密度。
推导和边界条件沿用 [原密度模块说明](VORONOI_DENSITY_GRAPH_TRANSFORMER_zh.md)。

## 2. 四项同维编码，直接拼接

设每路编码维度为 $d$，密度使用无偏置线性映射：

$$r_i=W_\rho s_i,\qquad W_\rho\in\mathbb R^{d\times1}.$$

$$c_i=\operatorname{Linear}(x_i)\Vert e_{v_i}\Vert\phi(t_i)\Vert r_i
\in\mathbb R^{4d}.$$

这里 $\operatorname{Linear}(x_i)$、$e_{v_i}$、$\phi(t_i)$、$r_i$ 各为 $d$ 维，
都属于同一个 X 观测点。$r_i$ 无偏置，因此 $s_i=0\Rightarrow r_i=0$。
标量扩维没有增加观测信息，也不保证提升效果。

第一层沿用 pre-norm attention，直接对拼接表示计算：

$$\bar c_i=\operatorname{LN}_{4d}(c_i),$$

$$q_i=W_Q\bar c_i+b_Q,\quad k_i=W_K\bar c_i+b_K,\quad
v_i=W_V\bar c_i+b_V,$$

$$W_Q,W_K,W_V\in\mathbb R^{d\times4d}.$$

**拼接与 Q/K/V 之间没有融合 MLP**，仅保留 attention 自身的 LayerNorm 和投影。
显式时间关系、变量关系及 attention 后原有 FFN 保留；不添加独立密度 logit。
密度只经节点内容进入 Q/K/V。

## 3. 残差与输出维度

拼接输入是 $4d$，第一层 attention 输出仍为 $d$。
为使残差相加维度一致，残差支路单独采用无偏置线性投影：

$$u_i=W_Rc_i+\operatorname{AttentionOutput}_i,\qquad
W_R\in\mathbb R^{d\times4d},$$

$$h_i=u_i+\operatorname{FFN}(\operatorname{LN}_d(u_i)).$$

该投影与 attention 并行，不是在 Q/K/V 前插入融合网络。
若有后续图层，其输入与输出均为 $d$，沿用原恒等残差。
X→Y 融合、各 backbone 的历史编码器、原预测头及各自损失不变。
默认 `plugin_hidden=32` 时，四路各 32 维，拼接为 128 维，Q/K/V 仍输出 32 维。

## 4. 去掉 s 的匹配消融

| 名称 | 输入密度段 | 拼接维度 | 拼接后融合 MLP |
| --- | --- | --- | --- |
| `voronoi_h_4d` | $W_\rho s_i$ | $4d$ | 无 |
| `voronoi_h_4d_no_density` | 全零 $d$ 维向量 | $4d$ | 无 |

消融不删除密度段、不缩小为 $3d$、不改变 Q/K/V 或残差投影尺寸。
两个版本的参数结构和参数数量相同；消融密度编码器仍保留，但因输入恒为零不参与有效学习。
规则采样时，两个版本在相同权重下输出严格一致。
这不等于要求整个节点表示为零，也不等于旧加和编码器。

与旧 `voronoi_h` 比较同时改变了输入维度、投影和融合结构，其效果差不能全归因于密度。
本次检验密度作用的主要对照是以上两个同结构版本。

## 5. 运行

```bash
git pull --ff-only origin master
# 运行 LatentODE 时需要 torchdiffeq；也可统一安装三个新增 backbone 的依赖
python -m pip install -r requirements-irregular-backbones.txt

# 每条命令运行一个模型的五个数据集，主版本加去掉 s 的消融，共十次
COMPARE=1 bash scripts/run_voronoi_h_4d_query_fusion.sh tpatchgnn --device cuda:1
COMPARE=1 bash scripts/run_voronoi_h_4d_query_fusion.sh hipatch --device cuda:1
COMPARE=1 bash scripts/run_voronoi_h_4d_query_fusion.sh mtand --device cuda:1
COMPARE=1 bash scripts/run_voronoi_h_4d_query_fusion.sh grafiti --device cuda:1
COMPARE=1 bash scripts/run_voronoi_h_4d_query_fusion.sh cru --device cuda:1
COMPARE=1 bash scripts/run_voronoi_h_4d_query_fusion.sh latentode --device cuda:1
```

不设置 `COMPARE=1` 时只跑主版本。默认只启动 t-PatchGNN 的五个数据集，
`all_six` 可顺序运行六个模型。固定 seed 3407，最多 30 epoch，保留原早停规则与配置 batch size。
支持指定单个数据集和显式 batch size：

```bash
BATCH_SIZE=16 COMPARE=1 bash scripts/run_voronoi_h_4d_query_fusion.sh hipatch physionet --device cuda:1
bash scripts/run_query_fusion.sh grafiti mimic voronoi_h_4d --device cuda:1
bash scripts/run_query_fusion.sh grafiti mimic voronoi_h_4d_no_density --device cuda:1
```

沿用 `PYTHON`、`TPATCH_DATA_ROOT`、`DIST_DATA_PATH`、`RUN_DIR`、`DRY_RUN=1`。
输出写入新的 `voronoi_h_4d_*` 目录，拒绝覆盖已有结果。
配置记录 `density_location=node` 和 `node_encoding=concat`；结果协议记录
`node_input_dim=4d`、`node_channel_dim=d`、`attention_hidden_dim=d`、
`pre_attention_fusion_mlp=false`，并标明密度通道是否启用。
