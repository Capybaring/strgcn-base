# Voronoi 密度关系 Graph Transformer（voronoi）

本方法实现讨论中确定的“内容、时间、变量、密度”四项并列 attention。
新方法名为 `voronoi`，避免与已有 `irgraph`（MLP 节点增量＋左右密度 Q/K 注入）实验混淆。
旧 `irgraph`、`strgcn`、历史结果和 backbone 代码保留原样；旧 checkpoint 不应用作新方法的 checkpoint。

## 节点密度

每个样本、每个 X 变量独立处理，对不同的真实观测时刻排序：

$$t_1<\cdots<t_n,\qquad \mu=\frac{t_n-t_1}{n-1}.$$

内部节点的 Voronoi 单元边界为左右邻点的中点，长度及密度为：

$$|V_i|=\frac{\delta_i^L+\delta_i^R}{2},\qquad
\rho_i=\frac1{|V_i|}=\frac2{\delta_i^L+\delta_i^R}.$$

每个单元承载一个单位观测计数，满足 $\rho_i|V_i|=1$。
在一维内部节点，该值也等于 DTFE 的节点密度估计。

首尾采用**对称延拓的边界约定**，令端点单元长度等于唯一可用的相邻间隔，$\rho_i=1/\delta_i$。
这不是直接截断到 $[t_1,t_n]$ 的 Voronoi 单元，否则规则端点的密度会加倍。
单个不同时刻没有可估计间隔，其结构编码为中性零；两个不同时刻都按单侧公式计算。
重复时刻共享几何量，所有原始观测 token/值仍然保留，估计对象是不同采样时刻的密度。

用于关系计算的是无量纲的相对对数密度：

$$s_i=\log(\mu\rho_i).$$

实际使用 float64 计算 $\log\mu+\log k_i-\log(\delta_i^L+\delta_i^R)$，内部 $k_i=2$、端点 $k_i=1$。
不先计算可能溢出的密度倒数，不裁剪 $s$，也不通过神经网络估计密度。
沿用旧实现的输入时间精度容差：当全部间隔偏差不超过
$2\epsilon_{\mathrm{input}}\max(\max_i|t_i|,\mu)$ 时，显式认定为可表示精度内的规则网格并置 $s=0$。
几何量不对时间戳求梯度。

## 独立的密度关系项

节点基础表示保持：

$$h_i^{(0)}=W_xx_i+e_{v_i}+\phi(t_i).$$

**不添加旧节点 MLP，不向 Q、K、V 单独注入密度。**
第 $a$ 个头的 X 自注意力为：

$$e_{ij}^{(a)}=
\frac{(W_Q\operatorname{LN}(h_i))^\top(W_K\operatorname{LN}(h_j))}{\sqrt{d_h}}
+\alpha_a\langle\phi(t_i)^{(a)},\phi(t_j)^{(a)}\rangle
+\beta_a\frac{\langle U_{v_i}^{(a)},V_{v_j}^{(a)}\rangle}{\sqrt{d_r}}
-\gamma_a(s_i-s_j)^2.$$

$$\gamma_a=\operatorname{softplus}(g_a)>0,$$

每层每头独立学习 $g_a$，初始化为 $\log(e-1)$，对应 $\gamma_a=1$。
后三项与内容项共同 softmax；密度项等价于乘性高斯相似度 $\exp[-\gamma_a(s_i-s_j)^2]$。
“相对密度相近更易交互”是显式建模假设，不是 Voronoi 理论给出的预测性能保证。

规则采样时各变量可以有不同频率，但均满足 $\mu\rho_i=1$，因此 $s_i=0$ 且密度关系严格为零，含首尾。
仅一端 $s=0$ 时仍允许非零关系：$b(0,s_j)=-\gamma_as_j^2$。
反过来，$s=0$ 不保证左右间隔均匀；例如 $(1,3)$、$\mu=2$ 与 $(2,2)$ 的密度相同。
不额外引入左右不对称度。

密度只作用于 X 图内部，**X→Y 沿用原 V1 交叉注意力，不加入密度项或密度 K 投影**。
没有给 Y 查询构造虚假的 X 采样几何。所有已提供 X 均可影响更早的 X/Y，未来 Y 真值从不作为输入。
历史编码器、预测头、内外生划分、训练归一化和评估协议均保持原样。

## 三批运行

在服务器的仓库根目录，先更新：

```bash
git pull --ff-only origin master
```

第一批 t-PatchGNN，第二批 Hi-Patch，第三批 mTAND；每批顺序跑五个数据集：

```bash
bash scripts/run_voronoi_query_fusion.sh tpatchgnn --device cuda:1
bash scripts/run_voronoi_query_fusion.sh hipatch --device cuda:1
bash scripts/run_voronoi_query_fusion.sh mtand --device cuda:1
```

固定 seed 3407、最多 30 epochs、FullExo、Y-only backbone 历史。
保留每个模型/数据集原配置的 batch size，不做自动降 batch；可显式设置 `BATCH_SIZE`。
每次创建全新结果目录，拒绝覆盖已有 `RUN_DIR`。
脚本默认只跑第一批 t-PatchGNN，不自动启动后两批。
`TPATCH_DATA_ROOT`、`DIST_DATA_PATH`、`PYTHON`、`RUN_DIR` 可显式设置，含空格路径受支持。
默认数据路径是 `./data/tpatch-data` 与 `./data/processed/DIST.npz`。

```bash
# 单数据集 / 单次实验
bash scripts/run_voronoi_query_fusion.sh tpatchgnn mimic --device cuda:1
bash scripts/run_query_fusion.sh hipatch mimic voronoi --device cuda:1
# 配置检查，不训练
DRY_RUN=1 bash scripts/run_voronoi_query_fusion.sh tpatchgnn --device cuda:1
# 同参数结构的密度关系消融，每批十次实验
COMPARE=1 bash scripts/run_voronoi_query_fusion.sh tpatchgnn --device cuda:1
```

`voronoi_no_density` 仅将结构编码 $s$ 置零，模型参数结构及初始化与 `voronoi` 相同。
它是判断密度关系作用的直接对照；不要用旧 `irgraph` 的 `no_node/no_density` 名称指代新方法。
协议记录在结果 `metrics.json` 中，配置及提交号记录在每次运行目录。
X 自注意力仍按接收节点分块并在训练时重计算，不永久存储额外完整密度矩阵，计算量仍为观测数平方。

## 数学来源

- Moradi et al. (2019), *Resample-smoothing of Voronoi intensity estimators*, Definition 1.
  https://doi.org/10.1007/s11222-018-09850-0
- Schaap & van de Weygaert (2000), *Continuous Fields and Discrete Samples: Reconstruction through Delaunay Tessellations*, Eq. (2).
  https://arxiv.org/abs/astro-ph/0011007
