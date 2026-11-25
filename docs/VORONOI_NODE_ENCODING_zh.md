# Voronoi 密度进入节点编码（voronoi_h）

本页保留原拼接后 MLP 版本。密度编码为同维向量、直接进入 attention 的新版本见
[voronoi_h_4d 与去掉 s 的消融](VORONOI_CONCAT_ENCODING_zh.md)。

本版本用于检验：密度进入节点内容是否比独立密度相似性 attention 偏置更适合当前任务。
不预设它一定提升精度。旧 `voronoi`（attention 偏置）、`irgraph` 和历史结果保持原样。

## 密度计算不变

按每个样本、每个 X 变量的不同观测时刻计算：

$$\rho_i=\frac{2}{\delta_i^L+\delta_i^R},\qquad s_i=\log(\mu\rho_i).$$

首尾仍用对称延拓的单边估计 $\rho_i=1/\delta_i$；孤立时刻无几何信息，$s_i=0$。
重复时间戳共享密度但所有观测 token 保留。
密度仍是数学公式的输出，不由网络估计。不增加不对称度，也不裁剪 $s$。

## 改动一：四路拼接得到 h

旧 `voronoi` 的节点编码是：

$$h_i^{(0)}=\operatorname{Linear}(x_i)+e_{v_i}+\phi(t_i).$$

新 `voronoi_h` 的输入为：

$$z_i=\operatorname{Linear}(x_i)\Vert e_{v_i}\Vert\phi(t_i)\Vert s_i,$$

$$h_i^{(0)}=W_2\operatorname{SiLU}(W_1z_i+b_1)+b_2.$$

前三路各为 $d$ 维，密度为 1 维，因此 $z_i\in\mathbb R^{3d+1}$。
融合网络为 Linear(3d+1,d) → SiLU → Linear(d,d)，无额外 dropout。
网络学习四路信息如何组合，密度值本身仍由 Voronoi 公式定义。
这是非线性融合，并非“拼接后一个线性层”等价的分路投影加和。

## 改动二：取消独立密度偏置

旧 `voronoi` 的图注意力为：

$$e_{ij}=e_{ij}^{\mathrm{content}}+b_{ij}^{\mathrm{time}}+b_{ij}^{\mathrm{var}}-\gamma(s_i-s_j)^2.$$

新 `voronoi_h` 恢复三个打分项：

$$q_i=W_Q\operatorname{LN}(h_i),\quad
k_j=W_K\operatorname{LN}(h_j),\quad
v_j=W_V\operatorname{LN}(h_j),$$

$$e_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}+b_{ij}^{\mathrm{time}}+b_{ij}^{\mathrm{var}}.$$

新模型不实例化旧左右密度 Q/K 投影，也不实例化密度偏置的 gamma 参数。
密度通过 h 影响当前层 Q、K、V，而不是预先要求“密度相近的节点更应交互”。
原有显式时间关系和变量关系不变；时间/变量既参与节点输入，也参与独立关系项，沿用现有设计。
X→Y 的公式不加新项，只接收更新后的 X 编码。
backbone 历史编码器、预测头和训练损失保持原样；未来 Y 标签不能进入输入，已提供未来 X 仍可影响更早查询。

## 对照与规则采样

- `voronoi`：旧的密度 attention 偏置版本。
- `voronoi_h`：新的四路节点融合，没有独立密度偏置。
- `voronoi_h_no_density`：与新版本**完全相同**的网络参数结构，仅将密度通道 $s$ 置零。

新旧主版本既改变了密度位置，也改变了节点融合结构，所以二者的性能差不能全部归因于密度位置。
`voronoi_h` 与 `voronoi_h_no_density` 才是检验新增密度通道作用的匹配对照。
规则采样时 $s=0$，新版本与相同权重的节点融合消融相同，但**不保证等于旧的加和编码器**。
两种输入编码的容量不同，不把规则采样的整个 h 强制归零。

## 分三批运行

```bash
git pull --ff-only origin master
# 第一批：t-PatchGNN × 五个数据集
bash scripts/run_voronoi_h_query_fusion.sh tpatchgnn --device cuda:1
# 第二批：Hi-Patch × 五个数据集
bash scripts/run_voronoi_h_query_fusion.sh hipatch --device cuda:1
# 第三批：mTAND × 五个数据集
bash scripts/run_voronoi_h_query_fusion.sh mtand --device cuda:1
```

默认 seed 3407、最多 30 epochs、FullExo、Y-only 历史，不改正式模型或 batch size。
默认只启动第一批，所有结果写入新的 `voronoi_h_*` 目录；可显式设置 BATCH_SIZE、TPATCH_DATA_ROOT、DIST_DATA_PATH、PYTHON、RUN_DIR。

```bash
# 单数据集
bash scripts/run_query_fusion.sh hipatch mimic voronoi_h --device cuda:1
# 同结构密度消融，每个模型十次实验
COMPARE=1 bash scripts/run_voronoi_h_query_fusion.sh tpatchgnn --device cuda:1
# 只生成配置
DRY_RUN=1 bash scripts/run_voronoi_h_query_fusion.sh tpatchgnn --device cuda:1
```

配置记录 `density_location=node`；结果 `metrics.json` 的 `protocol.fusion` 记录四路融合结构和密度使用位置。
