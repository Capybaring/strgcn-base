# t-PatchGNN 四数据集：批次条件、批量注意力和官方评估口径

本入口读取 `data/tpatch-data/{mimic,physionet,activity,ushcn}/processed/` 内的 `.pt` 记录，保留补充实验的“给定真实未来 X，预测未来 Y”设定。角色索引仍位于 `config/tpatch/roles/`，模型保持 **4层时间注意力＋3层跨变量注意力**，没有加入patch聚合或改变预测任务。

## 文件组织

```text
data/tpatch-data/
├── mimic/processed/mimic.pt
├── physionet/processed/set-a_0.0.pt
│                       set-b_0.0.pt
│                       set-c_0.0.pt
├── activity/processed/data.pt
└── ushcn/processed/ushcn.pt
```

`.pt` 的格式是 `list[(record_id, times, values, mask)]`，每条记录长度不同。三个PhysioNet集合先拼接，然后统一划分；不是分别作为训练/验证/测试。原始数据不下载、不重写、不在数据目录生成缓存。

## 与 t-PatchGNN 一致的任务与 batch 条件

[官方划分](https://github.com/usail-hkust/t-PatchGNN/blob/main/lib/parse_datasets.py)：先用种子42将80% seen/20% test随机划分，再将seen按原顺序分成前75% train/后25% valid。**源记录先划分，随后各自在记录内部构造任务**，不会将记录拼成全局长序列。

| 数据集 | 历史→预测 | batch_size（与官方运行脚本一致） |
|---|---|---:|
| MIMIC | 24→24小时 | 32 |
| PhysioNet | 24→24小时 | 32 |
| Activity | 3000→1000毫秒 | 32 |
| USHCN | 24→1个月 | 192 |

Activity仍按1000ms步长分段，USHCN在48个月内生成start=0..23的片段，保留原来的重叠、末端包含和边界规则。所有同源片段在同一集合。无历史或无未来有效目标的片段计数后跳过。Activity采用会话级划分以匹配基准，不改为按人分组；读取时仅在内存中稳定排序，保留重复时刻原观测。

[官方 batch 配置](https://github.com/usail-hkust/t-PatchGNN/blob/main/tPatchGNN/scripts/run_all.sh)。同样3,318个Activity训练片段，新默认是104步/轮，替代旧batch4的830步/轮。更大的batch并不自动保证8倍提速，下面的实现优化用于减少每步的小调用开销。

## 计算优化保持模型含义

新GraphTransformer配置启用 `dataset_params.pack_attention: true`。

- 在CPU collate中生成各变量的合法因果源/接收索引，避免前向对每个CUDA变量执行tolist/bool/int标量读取。
- 按关系类型和接收/源长度的2次幂区间分桶，将多个记录、多个变量的小注意力合成批量SDPA。
- `attention_score_budget` 默认1,048,576，限制每个桶每头的score位置数；超过时拆分，避免构造整个批次所有节点的方阵。
- 接收者分块时，仅保留该块最大合法前缀内的源；所有mask和时间先后关系与旧实现一致。
- 默认 `checkpoint_attention: false`，取消每个小注意力的反向重算；如显存紧张可在YAML开启，packed路径也支持checkpoint。
- 保留原始观测/查询节点、所有层数、Q/K偏置、FFN、参数结构和输出头。旧模型权重可用于等价核对，但训练尺度和指标已改变，建议新实验从头训练。

已有 `data/processed/*.npz` 的DAG配置没有切换评估协议；旧逐组注意力仍可作为核对路径。这里没有给旧DAG DataLoader隐式添加packed元数据。

## 新评估：不再使用 DAG 的 mse_norm

直接对齐 [t-PatchGNN lib/evaluation.py](https://github.com/usail-hkust/t-PatchGNN/blob/main/lib/evaluation.py) 的 `compute_error` 和 `evaluation`：

1. 对每个目标变量j，在整个验证/测试集合中累计平方误差、绝对误差和有效观测数。
2. 每个变量分别求平均误差，再对有观测的目标变量等权平均。
3. `RMSE = sqrt(最终的变量宏平均MSE)`，不是逐窗口RMSE平均。
4. 训练损失也按当前批次内每个变量先平均，再对可用目标变量平均，与upstream `compute_all_losses` 一致。
5. 早停和最佳模型按验证集 `mse` 选择。

默认输出 `mse`、`mae`、`rmse`，不再输出DAG的mse_norm，也不再拟合独立的StandardScaler或反归一化后评价。MAPE未列入本任务默认指标。

### 数值空间

- **MIMIC、PhysioNet、Activity：** 在seen源记录（train+valid，不含test）的有效观测上拟合逐变量MinMax；验证与测试在该空间计算指标。常数列范围按upstream设为1e-8。
- **USHCN：** 官方加载器对该已处理数据不再做MinMax；这里也采用恒等尺度，在文件提供的预处理空间计算指标。

这次为了按官方数值口径对齐，将先前本入口的“仅训练记录拟合尺度”改为了“seen记录拟合尺度”。测试数据从不参与尺度拟合。若训练/seen中某列没有有效值，保留恒等尺度并记录列号，不从test补统计。

**与原论文仍有明确的任务差异：** 当前只评价内生Y子集，并额外给定未来X；先前确定的PhysioNet无效负值/静态字段校验继续保留，不恢复已发现的异常占位。这些差异会改变有效mask和分数，因而“评估公式、数值空间与batch一致”不意味着与原论文全部变量预测的结果可直接比较。

## 变量和缺失值

| 数据集 | 动态未来X | 基线X | 默认目标Y |
|---|---|---|---|
| MIMIC | 42列药物/液体/摄入 | 无 | 52列生理化验/输出 |
| PhysioNet | FiO2、MechVent（14、24） | Age、Gender、Height、ICUType（0..3） | 35列 |
| Activity | CHEST_xyz、BELT_xyz（6..11） | 无 | 双踝xyz（0..5） |
| USHCN | SNOW、SNWD、PRCP（0..2） | 无 | TMAX、TMIN（3、4） |

MIMIC二分类内生54列中的52=Albumin、76=Pre-Admission仍因含义未决而默认排除，不作为输入或目标。MIMIC按核实的真实张量列顺序取索引，不能直接用LABEL_CODE。

PhysioNet只从有效time=0取基线；未来副本延用同一基线，不读取后续0占位，不从未来首次有效测量补齐。ICUType指入院类型。mask=1的真实0保留，包括Gender=0、MechVent=0及零坐标。负占位/负体温、无效性别、身高和ICUType按现有字段规则排除。

不规则观测不进行隐式插值；未来Y只作为监督标签，模型输入清零。未来X及其可用位置是oracle任务给定条件，不代表实际部署可提前获得，不作因果效应结论。

USHCN的列名顺序沿用[官方loader](https://github.com/usail-hkust/t-PatchGNN/blob/main/lib/ushcn.py)的声明；使用自行重新处理的文件时应核对Value_0..4的映射。

## 运行

```bash
# 新配置默认写入 *_tpatch_v2_oracle_x/，与旧DAG结果分开
DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh activity
DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh

# 去掉全部历史/未来X，写入 *_tpatch_v2_y_only/
NO_EXOGENOUS=1 DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh

# 可选：已有混合精度attention；默认FP32以核对工程等价性
ATTENTION_DTYPE=auto DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh activity

# 对同一真实批次在本机GPU测量旧loop+checkpoint和新packed路径
python scripts/benchmark_tpatch_packed.py --dataset activity \
  --data-root data/tpatch-data --device cuda:0
```

支持MODEL=ExEn、ExEnAligned、STRGCNDirect；它们的tpatch配置也同步了batch和评估，packed注意力优化只适用于ExEnGraphTransformer。支持TPATCH_DATA_ROOT、PYTHON、RUN_TAG。

结果中的预测和真实值均为**模型输入空间**；不要再按原来的DAG原始单位解释。模型目录保存scaler.npz和record_splits.npz；不会产生evaluation_scaler.npz。evaluation_protocol.json记录尺度、batch、目标、缺失规则和优化开关；测试导出record_id、window_id、native起点与mask以便复算。

数值检查覆盖新旧路径的预测/全参数梯度、checkpoint开关、不同时间嵌入、掩码与未来泄漏、变长补齐、整集合变量宏平均以及四套数据一轮CLI训练。服务器GPU的速度和峰值显存需用上面的脚本实测，CPU计时不作GPU加速承诺。
