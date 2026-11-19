# 方向 B：查询特征级外生融合

DIST三个基座×五种方法已接入：[DIST_QUERY_FUSION_zh.md](DIST_QUERY_FUSION_zh.md)。
入口：`bash scripts/run_dist_query_fusion.sh --device cuda:1`。

原始三基座的全变量训练／仅内生评估入口：
[ORIGINAL_BACKBONES_ENDOGENOUS_EVAL_zh.md](ORIGINAL_BACKBONES_ENDOGENOUS_EVAL_zh.md)。
一次运行：`bash scripts/run_original_backbones.sh`。

## 仓库核查与实施顺序

起点为 GitHub `master` 的 `22fa20e83e3a84ed6c6970f2cbbd6668d391182e`（2026-11-16 核对远端 HEAD）。最初按要求在独立开发分支实现；现按用户最新要求将实现同步到原 `master`，后续直接在原分支维护，不新增分支。原工作目录的四个未跟踪网格搜索/审计文件，以及 sources、原始数据和已有模型/结果保持原样。

## 服务器单次运行入口

在原仓库目录更新 `master` 后直接运行：

```bash
cd /home/ybhuo/hq/STRGCN
git pull --ff-only origin master
bash scripts/run_query_fusion.sh
```

默认只运行 **t-PatchGNN + Activity + base** 一次，固定随机种子3407，GPU 0，完整数据和默认模型结构，最多30轮/5轮早停。数据默认从当前仓库的 `data/tpatch-data` 读取。进度实时显示在终端，同时保存日志；每次创建带时间和进程号的新结果目录，结束时显示指标和保存位置。

每次指定一个基座、一个数据集、一个方法：

```bash
bash scripts/run_query_fusion.sh tpatchgnn activity base
bash scripts/run_query_fusion.sh tpatchgnn activity mlp
bash scripts/run_query_fusion.sh tpatchgnn activity cross
bash scripts/run_query_fusion.sh tpatchgnn activity v1
bash scripts/run_query_fusion.sh tpatchgnn activity v2
# 换用 Hi-Patch 或另一个数据集，同样只运行一次：
bash scripts/run_query_fusion.sh hipatch ushcn v1
```

支持的数据集：`activity mimic physionet ushcn`。额外消融方法：`v2_static v2_no_transfer v1_no_time v1_no_variable`。不要一次粘贴执行全部示例；按实际要运行的组合逐条执行。

可选设置如 `GPU=2 bash scripts/run_query_fusion.sh tpatchgnn activity v1`；其他数据路径可用 `TPATCH_DATA_ROOT=/path/to/tpatch-data` 指定。`DRY_RUN=1` 仅生成本次配置，不训练。原 Python 矩阵入口保留供后续评估使用，当前以本节单次入口为准，不执行多种子矩阵。

需要一次运行 Hi-Patch 的四个数据集、`base/mlp/v1` 三个方法时，使用：

```bash
bash scripts/run_hipatch_base_mlp_v1.sh
```

该入口在同一 GPU 上依次运行 Activity、MIMIC、PhysioNet、USHCN，每个数据集按 base、mlp、v1 的顺序训练，共12次，固定种子3407，完整数据，最多30轮和5轮早停。进度实时显示，每次调用生成新的输出根目录；根目录的 `summary.json` 随运行更新，全部结束后生成 `summary.csv`。某次训练正常返回非零退出码后，矩阵入口记录失败并继续下一组，最终返回非零状态；终端被中断时不会自动续跑。可用 `GPU=2` 指定显卡，或用 `DRY_RUN=1` 仅生成12组配置。

mTAN/mTAND 对应入口为 `bash scripts/run_mtand_base_mlp_v1.sh`，同样一次运行四个数据集的 base/mlp/v1 共12组。单次可用 `bash scripts/run_query_fusion.sh mtand activity v1`。该基座采用官方变分编码—解码器，训练保留一次潜变量采样和KL正则，评估使用潜变量均值；目标为现有宏平均预测MSE加0.001倍KL，并非原论文插值任务的完整IWAE目标。接口、固定源码及具体边界见 [mTAN说明](../baseline/mtan/README.md)。

核查发现：

- `baseline/patch_fusion/adapter.py` 的现有两个 Fusion 模型先输出标量预测，再做 DAG 两层 MLP；它们不是查询特征级融合。
- 两个 vendor 编码器最终都会拼接历史表示和查询时间编码，再调用原 `decoder`。只需提取这一处接口，无须替换历史编码算法。
- ExEnGraphTransformer 的 XY 层是内容点积加时间/变量偏置，使用物理时间因果掩码；原代码没有显式历史滞后状态迁移。V2 是本次新增的可检验研究假设。
- 四组 `.pt` 数据在本机可读，使用既有 `/private/tmp/tpatch-real-check` 只读链接。新运行不在数据目录写缓存。
- 当前测试环境只有 CPU，CUDA/MPS 均不可用；Hi-Patch 使用仓库既有 sum 消息聚合兼容实现，尚未验证原生 PyG 后端和 GPU 显存。

实施顺序：先锁定数据协议与原预测路径等价性；提取两个 `encode_queries`；实现共享 V1；以同一插件接入两个基座；增加 V2 和静态/无迁移消融；运行隔离测试、完整回归和真实数据诊断；最后逐个模型、数据集和方法进行正式训练，暂时固定一个种子。最后一步尚未完成，不能将诊断分数作为精度结论。

## 接口

```python
features = adapter.encode_queries(batch)  # [B,Q,Y,D_native]
features = adapter.fuse(features, batch)  # shape unchanged
prediction = adapter.decode(features)   # [B,Q,Y], original decoder
```

共享插件 `DynamicLagFusion.forward(features, query_times, query_ids, query_valid,
 sources, history_y=None, history_x=None)` 接收 `[B,Q_flat,D]`。`Observations` 只有有效观测值、物理时间、全局变量 ID 和观测 mask，没有监督标签参数。适配器将 Q×Y 展平，融合后恢复。D 可因基座而异，通过一个线性投影进入共享隐藏空间。

基座默认历史输入为 Y；X 列保留原变量编号/容量但值、时间、mask 全清零。插件独立处理原始 X 观测。`backbone_history: yx` 可以检验将历史 X 同时交给原编码器的设置。`base` 是同一 Y-only 编码器和原解码器，不代表重新复现论文全部变量的原任务。

数据节点角色沿用仓库：0=历史 X，1=历史 Y，2=给定未来 X，3=Y 查询。源必须是观测 X，且 `source_time <= query_time`；相同时刻允许传播。未来 Y 的值从不进入两个编码器或插件。真实零保留；padding 先清零再编码，空邻域严格返回输入特征。查询从不作为消息源。

## V1：无历史关系迁移

源观测编码为 value 线性编码 + 变量 embedding + 正弦时间编码。查询来自原基座的解码前特征。每个头使用：

`logit(q,s) = Q(q)·K(s)/sqrt(d_head) + gamma_t p(tq)·p(ts) + gamma_v U(y)·V(x)`。

掩码 softmax 后加权 V，经过输出投影作为查询特征残差。复用了原 ExEn XY 的打分形式和合法边定义；针对查询接口重新实现为独立模块，未搬入整套时序层、角色编码或 FFN。原解码器本身继续负责非线性预测。`cross` 去掉两种图偏置，保留相同源编码和掩码，作为普通交叉注意力对照。

## V2：历史滞后分布条件化

历史 Y 观测作为接收者，只关注不晚于它的历史 X。用 V1 的内容/图偏置打分得到历史注意力 a，再对真实时间差使用 K 个高斯基函数的归一化响应 phi：

`r[b,h,y,k] = mean_{historical receivers i of y with legal sources} sum_j a[b,h,i,j] * phi_k(t_i-t_j)`。

未来查询的注意力 logit 额外加入：

`gamma_lag[h] * log(eps + sum_k r[b,h,y,k] * phi_k(t_query-t_source))`。

无历史合法对时不添加迁移偏置。历史状态可微，不 detach；未来 X 只进入未来融合，不进入历史状态估计。未来条件化体现在每个查询/源的实际时间差，因此同一历史状态会随未来查询时间产生不同偏置。

边界：当前 r 为**样本/历史窗口条件化、按目标变量与头区分**的滞后摘要，X 变量在摘要中聚合；X/Y 变量对仍由 XY 变量偏置区分。它不是每个历史时刻都更新的递归状态，也不是完整逐 X-Y 对的滞后张量，更不是因果效应识别。若这些更强机制是研究要求，应作为后续独立版本，避免把其收益混入当前 V2。

`lag_max=1`、`lag_bins=8` 的单位是数据集归一化时间；滞后超出中心范围不会删边，只落到边缘基函数。四数据集物理单位不同，因此正式实验应同时报告归一化参数及换算后的物理滞后范围。默认时间缩放沿用各 packet，不擅自改任务。

## 数据与评价协议

全部沿用 `TPatchRecordProcessor`、`TPatchInputScaler`、`TPatchForecastTrainer`：

| 数据集 | 历史→预测 | 原始单位 |
|---|---|---|
| MIMIC、PhysioNet | 24→24 | 小时 |
| Activity | 3000→1000 | 毫秒 |
| USHCN | 24→1 | 月 |

先按源记录划分（split_seed=42；60% train / 20% valid / 20% test），再构造记录内部窗口。临床/Activity 在 seen=train+valid 上拟合 MinMax，test 不参与；USHCN 保留预处理尺度。只评价既定目标 Y 子集，保留原角色文件和无效值规则。全验证/测试集按变量累计误差，先求每变量均值再等权平均；RMSE 为最终宏平均 MSE 的平方根，早停按 valid MSE。这里不输出 DAG 的 `mse_norm`。详见原 `TPATCH_ORACLE_zh.md`。

给定未来 X 是 oracle 条件，不保证部署时能够提前取得。新入口暂不把 12 个 processed 数据集与四个记录数据集混为同一评估矩阵。

## 对照、消融与正式执行

矩阵共 2 基座 × 4 数据 × 9 变体 × 3 种子 = 216 次运行：

- `base`：原编码器和原解码器，无外生融合。
- `mlp`：原有 DAG 标量输出融合，因果前向填充 X；这是明确保留的输出级对照。
- `cross`：查询级普通交叉注意力。
- `v1`：查询级时间/变量偏置注意力，无历史迁移。
- `v2`：历史动态滞后分布条件化。
- `v2_static`：训练集共享的逐目标、逐头滞后分布，不读取历史关系。
- `v2_no_transfer`：移除历史迁移路径，应与 V1 等价。
- `v1_no_time`、`v1_no_variable`：分别移除图偏置。

可使用 `--input-mode HistExo/NoExo` 追加输入消融。HistExo 下插件仍接收历史 X；原标量 MLP 在该模式按已有协议将 X 向量置零，Y-only 基座本身不含历史 X，解释结果时需说明这一差别；可统一使用 `--backbone-history yx` 补充比较。

```bash
# 先导出全部具体配置，不训练。output 必须不存在。
python scripts/run_query_fusion.py --data-root /path/to/tpatch-data \
  --output results/direction_b_plan --plan-only

# 正式默认矩阵：各模型官方结构、30 轮/5 轮早停、三种子
python scripts/run_query_fusion.py --data-root /path/to/tpatch-data \
  --output results/direction_b_full_new --device cuda:0

# 第一阶段可单独执行
python scripts/run_query_fusion.py --data-root /path/to/tpatch-data \
  --output results/direction_b_stage1_new --device cuda:0 \
  --variants base mlp cross v1
```

输出根目录若已存在直接拒绝，模型/数据/种子/变体各自隔离，保留配置、日志、checkpoint、预测 mask、记录/窗口索引和协议。汇总记录 MSE/MAE、分配参数总数、各 epoch 训练时间与进程墙钟时间。训练时间包含批次准备、前后向和优化器，不含验证/测试；墙钟时间另含启动和数据读取。CUDA 时间有同步，不将进程墙钟误报为纯 GPU 算子时间。不默认报告未实测的显存数据。

`--small-model`、`--max-samples-per-split` 或小于30轮的运行均在 manifest 标记 diagnostic。正式统计应按数据集、基座、种子、输入协议逐一匹配，报告均值/标准差，V2 增益分别相对 V1/静态/无迁移比较；不要用 test 调参。

## 本地复现环境

本轮解释器为 `/private/tmp/rolling-graph-test-env/bin/python`；依赖版本见
`requirements-query-tested.txt`。该清单是现场环境记录；也可用原
`requirements-exen.txt` 加 `baseline/patch_fusion/requirements.txt` 安装依赖。

```bash
python -m pytest -q tests
```

现场验证记录和小规模矩阵汇总见 `reports/direction_b_validation.json` 及
`reports/direction_b_smoke.csv`；原始 checkpoint、数组和日志仅保存在新副本的
`results/direction_b_final_smoke_20261116/` 与
`results/direction_b_default_shape_20261116/`，不纳入 Git。

## 2026-11-16 验证结果

- 完整回归：357 项通过（其中新增查询融合测试 62 项）。
- 真实数据诊断矩阵：2 基座 × 4 数据 × 9 变体，72 次一轮训练均成功；隐藏维度8，每划分最多2个窗口，单种子3407。
- 默认结构检查：2 基座 × 4 数据 × V1/V2，16 次一轮训练均成功；使用默认编码器/插件维度，每划分最多2个窗口，不能代表默认完整 batch 的资源占用。
- 8 个基座/数据分组中，九种变体的测试标签、mask、变量、记录和窗口全部一致；V1 与 V2-no-transfer 的预测数组逐值完全一致。
- 216 组正式配置已生成，但全量、多种子训练未执行。这里的诊断分数不支持“插件提高精度”或“V2优于V1”的结论。
- 计算限制：接收者分块限制单块分数矩阵，历史 X/Y 两两注意力总计算及训练激活仍可能较大；完整临床 batch 的 GPU 显存和吞吐须单独测量。

## 新增：STRGCN 外生关系编码

`strgcn` 使用 STRGCN 的时间/变量关系与低秩 Chebyshev 编码 X，再使用现有
查询交叉注意力；允许已提供的未来 X 影响更早的 Y。旧方法保留。
实现与新协议、运行入口见 [STRGCN_QUERY_FUSION_zh.md](STRGCN_QUERY_FUSION_zh.md)。
