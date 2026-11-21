# 不规则采样增量 Graph Transformer（irgraph）

`irgraph` 是新的 X 编码器与查询级融合插件。t-PatchGNN、Hi-Patch、mTAND 的历史编码器、查询特征接口、原预测头和训练目标保持原样。X 编码由多头 Graph Transformer 完成，保留连续时间与变量关系偏置，随后沿用 V1 的 X→Y 交叉注意力。旧 `v1`、`v1_all` 和 `strgcn` 的算法及配置不改，历史结果不覆盖。

## 规则数据上为零的节点增量

只在同一样本、同一 X 变量内部，按真实观测时刻排序。设不同观测时刻为 $t_1<\cdots<t_n$，其参考间隔为

$$\mu_v=\frac{1}{n-1}\sum_{j=1}^{n-1}(t_{j+1}-t_j).$$

节点左右邻边存在标记为 $m_j^L,m_j^R$。不存在的边，其距离和标记均为零；存在的边使用实际时间差 $\delta_j^L,\delta_j^R$。定义

$$z_j^L=\frac{\delta_j^L-m_j^L\mu_v}{\mu_v},\qquad
z_j^R=\frac{\delta_j^R-m_j^R\mu_v}{\mu_v}.$$

将两个方向组成 $z_j$，边标记组成 $m_j$，节点增量为

$$\Delta h_j=F_\theta([z_j,m_j])-F_\theta([0,m_j]),\qquad
h_j^{(0)}=W_xx_j+e_{v_j}+\phi(t_j)+\Delta h_j.$$

$F_\theta$ 是 Linear→SiLU→Linear，允许学习偏置，但内部没有 dropout。使用同一个网络的两次计算相减，确保 $z=0$ 时所有学习偏置也完全抵消。规则采样时所有有效间隔等于该变量自己的 $\mu_v$；首尾缺边仍满足 $z=0$，因此增量恒为零。不同变量的规则采样频率可以不同。

只有一个不同时刻时不推断采样结构，所有几何量置零；两个不同时刻也无法推断间隔不均匀，增量为零。同一变量的重复时刻共享几何编码，**所有原观测值仍保留为独立 token**。计算不跨样本、不依赖 batch 统计，不人为添加首尾边。

时间差在 float64 中计算。如果所有间隔与均值的偏差不超过 `2 * 输入时间精度eps * max(最大时间绝对值, 平均间隔)`，将该序列识别为浮点舍入范围内的规则采样，显式置零。此容差只处理时间戳表示精度；低于输入精度的扰动无法可靠辨认。几何量不对时间戳求梯度，网络参数仍正常学习。

## 密度进入 attention 内容

定义左右相对密度

$$r_j^a=\begin{cases}\log(\mu_v/\delta_j^a),&m_j^a=1,\\0,&m_j^a=0.\end{cases}$$

计算后裁剪到 $[-8,8]$，避免极短间隔导致数值极端。相对密集为正，相对稀疏为负；规则采样为零。它是间隔的可解释变换，并非额外独立观测，也不预设“密集一定可信”。

每层 X 图注意力使用

$$q_i=W_Q\operatorname{LN}(h_i)+A_Qr_i,\quad
k_j=W_K\operatorname{LN}(h_j)+A_Kr_j,\quad
v_j=W_V\operatorname{LN}(h_j),$$

$$s_{ij}^{(a)}=\frac{(q_i^{(a)})^\top k_j^{(a)}}{\sqrt{d_h}}
+\tau_a\langle\phi(t_i)^{(a)},\phi(t_j)^{(a)}\rangle
+\gamma_a\frac{\langle U_{v_i}^{(a)},V_{v_j}^{(a)}\rangle}{\sqrt{d_r}}.$$

`phi` 使用已有 STRGCN 的连续时间 RBF 特征；变量偏置由可学习的低秩 U/V 分解得到。对所有合法 X 做 softmax，聚合 value，经输出投影、残差和前馈层更新。**不是 Chebyshev 图卷积**。没有新增跨变量的左右间隔边特征，时间几何仅来自同变量相邻观测；图注意力仍允许不同 X 变量交换信息。

最后 X→Y 交叉注意力还使用 $k_j^{XY}=W_K^{XY}h_j+A_{XY}r_j$，其余内容/时间/变量偏置沿用 V1。所有密度投影均不含偏置，$r=0$ 时严格无增量。Y 查询特征 G 不改结构，只在原预测头之前加融合残差。

在同一组权重下，规则采样的 `irgraph` 与关闭节点/密度增量的 `graph` 一致；**这不意味着整个插件等于零或等于 backbone**。原始的 X 值、时间、变量编码及交叉融合依然工作。该数学性质说明额外分支表达采样结构偏离，不能单凭该性质证明准确率更好、创新性或不规则数据优势，需要实测对照。

## 可见性、资源及评估

所有传入的历史与未来 X 均可影响任意 X 和任意 Y 查询，没有 `X_time <= Y_time` 限制。未来 Y 真值从未作为来源。未来 X 必须是预测时可获取的外部输入；这是当前 FullExo 的数据可见性协议。`HistExo`、`NoExo` 模式继续受数据入口控制。

三个 backbone、五个数据集沿用原 split、训练集归一化、内外生变量划分和仅内生变量的 macro MSE/MAE 评估。DIST 的数据处理也不变。X 图注意力按接收节点分块，并在训练时重计算块内 attention；计算量仍是观测数的平方，不代表修复了 Hi-Patch 自身的稠密邻接显存开销。脚本不更改默认 batch size（DIST=1，公开数据按各自配置，包括 USHCN=192），可自行指定 `BATCH_SIZE`。

## 运行及对照

更新后运行新方法，三个模型 × 五个数据集，固定 seed 3407，顺序执行：

```bash
git pull --ff-only
bash scripts/run_irgraph_query_fusion.sh --device cuda:1
```

单个模型的五个数据集：

```bash
bash scripts/run_irgraph_query_fusion.sh hipatch --device cuda:1
```

单次实验、可选 batch size：

```bash
bash scripts/run_query_fusion.sh hipatch mimic irgraph --device cuda:1
BATCH_SIZE=1 bash scripts/run_query_fusion.sh hipatch mimic irgraph --device cuda:1
```

加入对照及消融（可选择一个或全部数据集）：

```bash
COMPARE=1 bash scripts/run_irgraph_query_fusion.sh hipatch physionet --device cuda:1
```

| 方法 | X 编码 | 节点增量 | 密度 attention | X 时间可见性 |
|---|---|---|---|---|
| v1 | 单点线性/变量/时间 | 无 | 无 | 不晚于查询 |
| v1_all | 同 v1 | 无 | 无 | 全部已提供 X |
| strgcn | 原 STRGCN 中间层图卷积 | 无 | 无 | 全部已提供 X |
| graph | Graph Transformer | 关闭 | 关闭 | 全部已提供 X |
| irgraph_no_node | 同上 | 关闭 | 开启 | 全部已提供 X |
| irgraph_no_density | 同上 | 开启 | 关闭 | 全部已提供 X |
| irgraph | 同上 | 开启 | 开启 | 全部已提供 X |

`COMPARE=1` 跑表中全部七项，包括旧的 `v1`（linear）和允许全部 X 的线性编码对照 `v1_all`，排除时间可见性不同造成的混淆。四个 Graph Transformer 变体实例化完全相同的参数结构，以便相同 seed 下初始化一致。

`TPATCH_DATA_ROOT` 默认 `./data/tpatch-data`；`DIST_DATA_PATH` 默认 `./data/processed/DIST.npz`；`DRY_RUN=1` 只生成配置。每次新建结果目录，拒绝覆盖已有目录，生成的配置、manifest、训练日志、预测和 summary 用于审查。正式多轮 GPU 实验应在服务器完成，本地短程检查仅验证可运行性，不作为性能结论。
