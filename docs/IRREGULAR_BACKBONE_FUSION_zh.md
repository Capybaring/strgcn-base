# GraFITi / CRU / LatentODE 的 Voronoi 节点密度融合

本次新增 `GraFITiQueryFusion`、`CRUQueryFusion`、`LatentODEQueryFusion`。
使用已有 `voronoi_h` 插件，不改变其密度公式、X 图编码器与 X→Y 融合机制。
三个模型均提供 Activity、MIMIC、PhysioNet、USHCN、DIST 配置。
原模型算子来源、固定版本和兼容性修改见
[backbone 来源说明](../baseline/irregular_backbones/README.md)。

## 融合位置

统一流程为：历史 Y → backbone 查询表示 → 外生 X 融合 → 原始预测头。
完整历史编码器继续使用各作者实现，未来 Y 真值仅用于损失和评估。

| Backbone | 插件输入 | 融合后使用的原预测头 |
| --- | --- | --- |
| GraFITi | 每个查询时间、目标变量的图边表示，维度 `3 * hid_dim` | 原 `Linear(3 * hid_dim, 1)` |
| CRU | 查询时刻连续滤波隐状态均值，维度 `latent_dim` | 原三层均值解码网络和输出层 |
| LatentODE | 查询时刻生成 ODE 潜在状态，维度 `latent_dim` | 原 `Linear(latent_dim, num_vars)` 的对应目标输出行 |

CRU 和 LatentODE 的共享状态先沿目标变量展开，插件结合查询变量 ID 产生不同的
目标表示，再交给原头相应输出行。CRU 的协方差路径保留原滤波结果和完整正方差头，
只对均值路径融合 X；连续状态保持 float64，插件使用自身 dtype 后再转回解码 dtype。
LatentODE 从历史首个时间点的后验初始状态积分到未来查询，训练采样，评估取后验均值。

`MTANQueryDataset` 仅复用长表到带 mask 历史序列的整理，不代表使用 mTAN 模型。
不同样本分别运行原编码器，去掉 padding 历史及 padding 查询，避免填充改变图结构或动力学。
重复时间戳的同变量观测保留为独立观测行，不求和，不修改数据文件。

## 密度插件沿用当前版本

X 观测节点按同变量计算一维 Voronoi 密度：内部点

$$\rho_i=\frac{2}{\delta_i^L+\delta_i^R},\qquad s_i=\log(\mu\rho_i).$$

首尾对称延拓下使用 $\rho_i=1/\delta_i$；单独一个时刻令 $s_i=0$。
每个样本、每个 X 变量独立计算，规则采样的 $s_i=0$。

$$h_i=\operatorname{MLP}\bigl([\operatorname{Linear}(x_i)\Vert e_{v_i}\Vert\phi(t_i)\Vert s_i]\bigr).$$

之后进入原有 X Graph Transformer 和查询融合，不额外增加密度 attention 偏置。
默认 `FullExo` 允许未来 X 影响更早的 Y 查询，不读取未来 Y。
完整公式及边界条件见 [Voronoi 节点编码](VORONOI_NODE_ENCODING_zh.md)。

## 训练目标与对照

- GraFITi：沿用本仓库变量宏平均预测 MSE。
- CRU：变量宏平均高斯 NLL，原方差头参与训练。
- LatentODE：变量宏平均预测 MSE 加 `kl_weight * KL(q(z0)||N(0,I))`，默认权重 `0.001`。
  这是统一预测任务适配，未使用原论文完整的重要性加权似然训练流程。

三个模型均按验证 MSE 早停，最终报告原有口径的 MSE/MAE。
配置是本仓库预测协议下的起始配置，不能把结果直接等同于原论文表格的复现。
建议每个 backbone 比较 `base`、`voronoi_h_no_density` 和 `voronoi_h`：
前者检验整体插件，后两者保持相同节点融合容量，仅对照密度通道。

## 安装与三批运行

在服务器仓库根目录执行：

```bash
git pull --ff-only origin master
python -m pip install -r requirements-irregular-backbones.txt

# 第一批：GraFITi，五个数据集
bash scripts/run_irregular_voronoi_h.sh grafiti --device cuda:1

# 第二批：CRU，五个数据集
bash scripts/run_irregular_voronoi_h.sh cru --device cuda:1

# 第三批：LatentODE，五个数据集
bash scripts/run_irregular_voronoi_h.sh latentode --device cuda:1
```

每条命令串行运行对应模型的五个数据集；seed 固定 3407，最多 30 epoch，早停 patience=5。
默认 batch size 与已有同数据集配置一致：Activity/MIMIC/PhysioNet 为 32，USHCN 为 192，DIST 为 1。
不自动减小 batch size。各样本独立求图/ODE/滤波可能增加运行时间，正式 GPU 耗时待实测。

单独指定数据集和 batch size：

```bash
BATCH_SIZE=16 bash scripts/run_irregular_voronoi_h.sh grafiti physionet --device cuda:1
BATCH_SIZE=32 bash scripts/run_irregular_voronoi_h.sh cru ushcn --device cuda:1
bash scripts/run_query_fusion.sh latentode mimic voronoi_h --device cuda:1
```

整批基线和密度消融（每模型 15 次实验）：

```bash
COMPARE=1 bash scripts/run_irregular_voronoi_h.sh grafiti --device cuda:1
COMPARE=1 bash scripts/run_irregular_voronoi_h.sh cru --device cuda:1
COMPARE=1 bash scripts/run_irregular_voronoi_h.sh latentode --device cuda:1
```

可用 `PYTHON` 指定解释器、`TPATCH_DATA_ROOT` 指定公开数据目录、`DIST_DATA_PATH`
指定 DIST 文件、`RUN_DIR` 指定新的输出目录。默认自动创建新目录，已有目录会拒绝覆盖。
`DRY_RUN=1` 只生成计划和配置，仍检查数据路径存在。
依赖新增 `torchdiffeq`；默认完整 CRU 无需 `geotorch`。
