# t-PatchGNN / Hi-Patch + DAG MLP Fusion

DIST 已支持：`BACKBONE=tpatchgnn DEVICE=cuda:0 bash scripts/run_patch_fusion.sh DIST`，Hi-Patch 使用 `BACKBONE=hipatch`。三个模型统一运行：`DEVICE=cuda:0 bash scripts/run_dist.sh`。数据与新评估见 [DIST 说明](../../docs/DIST_FORECAST_zh.md)。无参数仍只跑原四个数据集。

这两个比较方法保留原始历史编码骨干，在预测端严格采用DAG的通用两层MLP融合结构：

```text
Z = Backbone(Y_history, X_history; future_query_times)[..., Y_columns]
Y_pred = Linear_2(ReLU(Linear_1(Concat(Z, X_future_at_query))))
```

默认MLP隐藏宽度64，输入宽度DY+DX、输出DY，没有残差、交叉注意力、时间差/mask拼接或其他专用外生分支。骨干和MLP一起由Y上的变量宏平均MSE训练。模块与DAG参考版本 `0758990e2c73bb54138ea3e7b11a35cbc5476bcc` 的 `ts_benchmark/baselines/utils.py::MLP` 做过相同权重下的输出和参数梯度对照。

这是“遵循DAG基线适配的扩展模型”，不是t-PatchGNN/Hi-Patch原论文无需改造的原版结果，也不是DAG整套实验的复现。

## 安装和数据

```bash
python -m pip install -r baseline/patch_fusion/requirements.txt
```

原数据仍放在 `data/tpatch-data/{mimic,physionet,activity,ushcn}/processed/`，与[四数据集说明](../../docs/TPATCH_ORACLE_zh.md)一致。数据划分、临床/Activity MinMax seen拟合、USHCN原预处理空间、目标mask和整集合变量等权MSE/MAE/RMSE均复用当前TPatchForecastTrainer，不使用DAG评估指标。

角色文件继续使用 `config/tpatch/roles/`。MIMIC目标52列（排除52=Albumin、76=Pre-Admission两项待核实变量），X为42列；PhysioNet有35个目标和6个条件，其中4个静态基线；Activity为双踝6目标、胸部/腰带6条件；USHCN为最高/最低温2目标、3个气象条件。

## 运行

```bash
# 两个模型 × 四个数据集，默认FullExo
DEVICE=cuda:0 bash scripts/run_patch_fusion.sh

# 只运行一个骨干和一个数据集
BACKBONE=tpatchgnn DEVICE=cuda:0 bash scripts/run_patch_fusion.sh activity
BACKBONE=hipatch DEVICE=cuda:0 bash scripts/run_patch_fusion.sh activity

# 相同容量的历史X、无X对照
BACKBONE=tpatchgnn MODE=HistExo DEVICE=cuda:0 bash scripts/run_patch_fusion.sh activity
BACKBONE=tpatchgnn MODE=NoExo DEVICE=cuda:0 bash scripts/run_patch_fusion.sh activity
```

BACKBONE可为all（默认）、tpatchgnn、hipatch；MODE可为FullExo（默认）、HistExo、NoExo。支持TPATCH_DATA_ROOT、PYTHON、RUN_TAG。各模型/模式的模型目录及结果目录自动分开，RUN_TAG自定义时仍保留模型子目录。

也可直接通过main.py使用以下配置：

```bash
python main.py --config_path config/TPatchGNNFusion/tpatch/activity.yaml \
  --device cuda:0 --model_save_dir_path model_states/tpatch_fusion_activity \
  --result_save_dir_path results/tpatch_fusion_activity
```

Hi-Patch对应 `config/HiPatchFusion/tpatch/activity.yaml`。直接运行时可添加 `--history-only-exogenous` 或 `--no-exogenous`，两者互斥。

| 模式 | 骨干历史输入 | 融合层的未来X输入 |
|---|---|---|
| FullExo | 有效历史Y和X | 对齐后的给定未来X |
| HistExo | 有效历史Y和X | 全0，不读取真实未来X |
| NoExo | 只有有效历史Y，X列mask=0 | 全0，不读取任何X |

三种模式保留同样的骨干变量维度和同样的MLP参数量，避免将“多了一个预测头”的容量收益归为未来X收益。因此HistExo/NoExo是**带相同MLP容量的适配模型消融**，不是宣称完全复现DAG仓库的无协变量原版设置。NoExo中X占位列只有固定0与缺失mask，不含患者/样本的X观测值。

## 不规则数据如何适配

- CPU批次整理时，将原始有效历史观测按骨干各自patch大小打包为 `[B,M,Lmax,N]`，保留观测时间和mask。一次分组打包，避免逐样本×patch×变量反复扫描。
- 骨干只编码历史Y/X。不会将未来X塞入历史patch，也不将未来Y的值用作插值或输入。未来目标的查询时刻是任务给定位置。
- 为每个样本收集实际Y查询时刻，骨干在这些时刻预测全部Y，再与同一时刻的X向量拼接。评价仍只取真实有效的Y标签位置。
- 不规则未来X使用统一的因果前向填充：在预测时刻t，取不晚于t的最近有效X。若给定未来窗口中该列较早位置无记录，可沿用该窗口历史已有值；如果连历史也没有，填0。不用t之后的值反向插值。
- HistExo直接将融合输入置0，由骨干负责利用历史X；NoExo同时删去历史X。FullExo的静态X只使用已验证基线及其固定副本。
- 融合MLP不增加mask或delta特征，以遵循DAG模块；因此“无值填0”与“真实0”的区别不会额外作为融合特征提供。这是将稠密模块用于稀疏数据的数据适配约定，需在论文中报告。

原始变量全局编号不变。骨干局部变量顺序为 `target_ids + sorted(exogenous_ids)`，输出先取Y部分。整个流程不会改写原始`.pt`。

## 默认骨干参数

默认patch大小、hidden维度、GNN/Transformer层数和头数来自各方法相应官方运行脚本；没有把两个方法强行替换成同一个骨干。为了沿用已统一的比较协议，batch统一为临床/Activity 32、USHCN 192；这与部分Hi-Patch原脚本较小batch不同。

| 数据集 | t-PatchGNN：patch/hidden/层/头 | Hi-Patch：patch/hidden/层/头 |
|---|---|---|
| MIMIC | 8小时 /64/1/1 | 12小时 /128/1/1 |
| PhysioNet | 8小时 /64/1/1 | 6小时 /64/1/1 |
| Activity | 300ms /32/1/1 | 750ms /64/3/1 |
| USHCN | 2个月 /32/1/1 | 1.5个月 /64/2/4 |

Hi-Patch保留其层次图算法，临床数据较密集时其图结构可能需要较大显存；本地只验证了小规模CPU前后向，尚未在你的GPU上验证默认完整batch的峰值显存。若需因显存调整batch，应记录配置差异，不能将不同batch下的epoch耗时直接视为同条件效率。

MLP宽度64一致；新增参数数目为 `(DY+DX)*64+64+64*DY+DY`，已包含于模型参数和优化器中。默认lr=0.001，保持当前统一的30轮/5轮早停预算；学习率在这组配置中保持常数。正式对比应报告参数来源和额外调参预算，建议用多个种子检查稳定性；也可明确采用官方推荐配置。

## 骨干来源与兼容修改

`vendor/PROVENANCE.json` 固定官方提交和源文件哈希：

- t-PatchGNN：`00c94e7bbaf21c71b03ed84ff690ae59e37129e5`。只删除未用的导入，将两处硬编码CUDA分配改为配置device。
- Hi-Patch：`ae3cb7c672e045db4b11085251a60b0c33b8d0db`。保留原消息、注意力、层次聚合。补齐B=1时被无约束squeeze删除的维度，并处理空图softmax。
- Hi-Patch优先使用安装的PyG/torch-scatter；缺少时使用本目录compat.py的gather→原message→index_add(sum)→原update后备实现。后备不是新图网络，测试覆盖求和结果及梯度；当前本机没有PyG，未做原生PyG运行的现场对照。

没有复制本项目原生ExEn的关系模块到比较方法中。新增MLP遵循DAG，原有t-PatchGNN/Hi-Patch不直接获知未来Y。

## 检查

```bash
python -m pytest -q tests/test_patch_fusion.py
```

覆盖：官方式两层MLP（无残差）、两个骨干×四数据×三模式前后向、全部历史观测打包保真、空历史、有效零、未来Y标签隔离、晚于查询的X隔离、8组CLI一轮训练和结果导出。另使用本机同版真实处理数据对两个骨干的四组默认结构各做了一次小规模前向/反向检查，不代表完整GPU训练或性能结论。
