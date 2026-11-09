# t-PatchGNN 四数据集：记录批次与给定未来 X 的补充实验

这一路径直接读取 t-PatchGNN 已处理 `.pt` 文件，服务于此前确定的 **oracle 条件预测补充实验**：给定历史 X/Y 与真实未来 X，预测未来 Y。它不宣称临床治疗、身体坐标或气象实测在现实中提前可得，也不声称本配置与 t-PatchGNN 原论文使用相同信息条件。

## 文件放置

```text
data/tpatch-data/
├── mimic/processed/mimic.pt
├── physionet/processed/set-a_0.0.pt
│                       set-b_0.0.pt
│                       set-c_0.0.pt
├── activity/processed/data.pt
└── ushcn/processed/ushcn.pt
```

每个 `.pt` 是 `list[(record_id, times, values, mask)]`，values/mask 的形状为 `[T_i,D]`。原始数据与压缩包不必移动；本入口不下载或重新处理原文件，不在数据目录写缓存。三个 PhysioNet 集合先按 a、b、c 拼接，**不是**分别作为训练/验证/测试集。

## 划分与任务边界

严格采用 [t-PatchGNN parse_datasets.py](https://github.com/usail-hkust/t-PatchGNN/blob/main/lib/parse_datasets.py) 的两步划分：

1. `train_test_split(records, train_size=.8, random_state=42, shuffle=True)`。
2. 对 seen 部分 `train_test_split(seen, train_size=.75, random_state=42, shuffle=False)`。
3. **完成源记录划分后**，在每个集合的记录内部构造任务。

| 数据 | 源记录 | 历史 → 预测 | 记录内任务 |
|---|---|---|---|
| MIMIC | 患者/住院序列 | 24 → 24 小时 | 每条记录一个任务，历史 `<24`，未来 `>=24`，上限 48 |
| PhysioNet | ICU 序列 | 24 → 24 小时 | 同上 |
| Activity | 活动会话 | 3000 → 1000 毫秒 | 以 1000ms 步长构造约 4000ms 任务；遵循 upstream `range(0,int(t_max)-history,pred_window)` 和最后片段包含末端规则 |
| USHCN | 站点记录 | 24 → 1 个月 | 在 48 个月记录中，start=0..23，每月一个任务，最后片段包含月 48 |

**这不是把所有患者/会话/站点拼成一条长序列再滑动划分。** 但 Activity 与 USHCN 在同一源记录内部的任务仍会重叠，这是 t-PatchGNN 的任务构造方式，不能为了“批次化”而删掉它们。同一源记录的所有重叠片段始终在同一集合。

无历史观测或无有效未来目标的片段会被跳过并统计；这会令任务数少于机械切片数。Activity 源数据可能存在时间戳回退或重复，加载后在内存中稳定排序、保留重复时刻的原观测，保证索引切分正确；不添加插值或不明来源的平均值。

Activity 与基准一致是**会话级**划分，并非按五位参与者做留人测试；同一人的不同会话可能出现在不同集合。本入口没有擅自改成按人分组。USHCN 在站点级划分后构造片段。

## 变量角色

`config/tpatch/roles/*.indices.json` 保存完整原始列顺序、0-based 选列索引与 0/1 标签，不按变量字典编号重新排序。

| 数据 | 动态未来条件 X | 静态条件 | 监督目标 Y |
|---|---|---|---|
| MIMIC | 42 列给药/液体/血制品/摄入记录 | 无 | 52 列生理化验/实际输出 |
| PhysioNet | `[14,24]`：FiO2、MechVent | `[0,1,2,3]`：Age、Gender、Height、ICUType | 35 列动态观测 |
| Activity | `[6..11]`：CHEST_xyz、BELT_xyz | 无 | `[0..5]`：ANKLE_LEFT_xyz、ANKLE_RIGHT_xyz |
| USHCN | `[0,1,2]`：SNOW、SNWD、PRCP | 无 | `[3,4]`：TMAX、TMIN |

MIMIC 二分类索引中 54 列标为内生，其中 `[52,76]`（Albumin、Pre-Admission）的含义仍待核实；处理器明确排除这两列，不作为输入或目标，因此默认训练目标是 52 列，既不丢失原始索引，也不把未确定项目静默纳入训练。

PhysioNet 静态条件仅从有效 `time=0` 值提取。在历史起点加入基线，在预测起点加入同一固定基线的副本表示未来条件，所有查询可访问它；不读取未来稀疏四列的 0 占位，也不把后来测到的身高回填为基线。基线缺失则不创建该条件节点。这里 ICUType 始终指入院时类型。

USHCN 名称顺序采用 [官方 loader](https://github.com/usail-hkust/t-PatchGNN/blob/main/lib/ushcn.py) 文档声明；如果使用自行重处理的不同 CSV，需核对 Value_0..4 的实际含义后修改角色映射，不应只按维数相同套用。

## 批次、掩码与标准化

`TPatchRecordDataset` 惰性生成一条记录内任务的 long-format 观测节点，DataLoader 只填充到**当前批次**的最大节点数；不把所有记录展开成一个全局矩阵，不提前存储所有重叠片段的复制数据。样本保留不规则时间和逐变量缺失掩码。

- 输入包括有效历史 X/Y、有效未来 X；未来 Y 由预测掩码标记，模型/训练器输入时清零，仅用于监督和评估。
- mask=0 的位置不创建节点；**mask=1 的真实 0 是有效值**，与现有 `data/processed/*.npz` 的零值规则分开。
- PhysioNet 负占位/负体温不作为有效观测；Gender 限于 0/1、Height 必须正值、ICUType 限于 1..4；不会把 Gender=0 删除。其他更细的临床异常值规则没有隐式追加。
- 输入 MinMax 和评估 StandardScaler 均仅拟合**训练源记录**的有效观测，每条训练观测只计一次，不因重叠片段多次计权。未在训练中出现的变量用恒等尺度，记录其索引，不使用验证/测试数据补统计。
- 输入时间：临床以 48 小时归一化；Activity 以片段相对时距/4000ms；USHCN 保留记录内绝对月时距并除以 48，以保留季节位置，符合 upstream 对 t_bias 的处理。

这里对齐的是 t-PatchGNN 的划分和预测时长，**不复制其用 train+valid(seen) 计算尺度的做法**。评估继续使用本仓库现有 `ExEnDAGForecastTrainer`：训练集 StandardScaler 空间的 mse_norm 等，以及原始输入单位指标，按有效目标/窗口聚合。数据若已标准化，“原始输入单位”不代表物理单位。不能把这些数字当作原论文相同归一化与聚合方式的复现。

## 运行

默认接入当前 ExEnGraphTransformer（沿用 4+3 层等模型参数），并为 ExEn、ExEnAligned、STRGCNDirect 同时提供独立配置，原有连续数据配置不变。

```bash
# 四个数据集，默认 ExEnGraphTransformer
DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh

# 单独运行一个或多个
DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh physionet activity

# 相同目标，移除全部历史/未来 X 的 Y-only 对照
NO_EXOGENOUS=1 DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh

# 选择其他已接入骨干或数据路径
MODEL=STRGCNDirect TPATCH_DATA_ROOT=/path/to/tpatch-data DEVICE=cuda:0 \
  bash scripts/run_tpatch_oracle.sh

# 有 CUDA 时可启用已有的混合精度 attention（仅 GraphTransformer）
ATTENTION_DTYPE=auto DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh
```

也可直接运行：

```bash
python main.py --config_path config/ExEnGraphTransformer/tpatch/physionet.yaml \
  --device cuda:0 --tpatch-data-root ./data/tpatch-data \
  --model_save_dir_path model_states/physionet_oracle \
  --result_save_dir_path results/physionet_oracle
```

运行器支持 `PYTHON`、`RUN_TAG`；FullExo 和 Y-only 默认输出目录不同。已有 conda 环境依照 `requirements-exen.txt` 使用，无新增必需依赖。仅加载 tensor/primitive 类型的 `.pt`，不回退到任意 pickle 执行；若自定义处理文件含其他对象，应先确认格式再转换。

若要只去掉未来 X 而保留历史 X，将对应 YAML 的 `known_future_exogenous_var_ids` 设为 `[]`，并选择独立的输出目录。不要将其与 `NO_EXOGENOUS=1` 混淆，后者同时移除历史 X。

## 输出与检查

模型目录包含 `record_splits.npz`（分段前的源记录分配）、训练输入 scaler、评估 scaler、模型及运行配置。结果目录新增：

- `test_record_id.npy`：每个预测片段对应的源记录 ID。
- `test_window_start_native.npy`：同序片段起点；单位在 evaluation_protocol.json 中记录，不误标为秒。
- 原有预测、真实值、mask、变量编号和 window_id，以及 metrics.json。

`evaluation_protocol.json` 同时记录划分种子、实际源文件、角色文件、排除列、时间单位、oracle 条件、缺失处理和训练尺度范围。

```bash
python -m pytest -q tests/test_tpatch_records.py tests/test_exen_processed.py \
  tests/test_dag_evaluation.py tests/test_no_exogenous.py
```

测试覆盖两步源记录划分、全部时长及边界、时间戳回退/重复、零值和无效静态值、未来条件与目标隔离、四种骨干的前向/反向、四种数据的一轮 CLI 训练及输出协议。原始数据不会随代码提交。
