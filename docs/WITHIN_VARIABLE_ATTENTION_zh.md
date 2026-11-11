# 时间与组内变量模块消融

ExEnGraphTransformer 按顺序执行三个阶段。

1. 单变量时间注意力，连接同一个变量不晚于接收时刻的有效观测。
2. 组内变量注意力，分别建立 X→X 和 Y→Y，连接同组不同变量的不晚于接收时刻的有效观测。
3. 原有 X→Y 注意力，把可用外生信息传递给内生节点。

所有源节点均为已观测的历史 Y 或提供的 X，未来 Y 查询只作接收节点。新模块有独立 Q/K/V、时间/变量偏置、FFN 和层归一化。关闭一个组时，该组的表示跳过整个新层，包括 FFN。

当前 17 份 ExEnGraphTransformer 数据集配置默认采用时间 4 层、组内变量 1 层、X→Y 3 层。这是在原来的 4+3 层之间新增 1 层，参数量与运算量会增加。若要固定总深度，可以比较 4+0+3 和 3+1+3。组内层数为 0 的模型可以严格加载原模型 state_dict。

```yaml
model_config:
  model_params:
    n_layer: 4
    n_variable_layer: 1
    n_cross_layer: 3
    within_exogenous: true
    within_endogenous: true
```

时间、组内变量、X→Y 的层数均允许为 0。`n_variable_layer: 0` 关闭整个组内模块。`within_exogenous` 和 `within_endogenous` 分别控制两组，两者同时为 false 时新模块整体跳过。

## 命令

Activity 上运行自己的模型。末尾可换成 mimic、physionet、ushcn，也可省略以顺序运行四个数据集。

```bash
# 完整模型
TIME_LAYERS=4 VARIABLE_LAYERS=1 WITHIN_VARIABLES=both DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh activity
# 关闭组内变量模块，恢复原 4+3 结构
VARIABLE_LAYERS=0 DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh activity
# 只开放 X 组内部跨变量边
VARIABLE_LAYERS=1 WITHIN_VARIABLES=exogenous DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh activity
# 只开放 Y 组内部跨变量边
VARIABLE_LAYERS=1 WITHIN_VARIABLES=endogenous DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh activity
# 关闭时间模块
TIME_LAYERS=0 VARIABLE_LAYERS=1 DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh activity
# 固定总深度的对照
TIME_LAYERS=3 VARIABLE_LAYERS=1 DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh activity
```

DIST 对应脚本如下，支持相同模块开关。

```bash
MODEL=exen TIME_LAYERS=4 VARIABLE_LAYERS=1 WITHIN_VARIABLES=both DEVICE=cuda:0 bash scripts/run_dist.sh
MODEL=exen VARIABLE_LAYERS=0 DEVICE=cuda:0 bash scripts/run_dist.sh
MODEL=exen TIME_LAYERS=0 VARIABLE_LAYERS=1 DEVICE=cuda:0 bash scripts/run_dist.sh
```

其他 processed 数据集可用 `scripts/run_exen_graph_transformer.sh`，支持相同开关。
`CROSS_LAYERS=0` 可以关闭 X→Y 阶段。四数据集的 `NO_EXOGENOUS=1`、DIST 的 `MODE=HistExo/NoExo` 仍可组合使用。

提供任一模块环境变量时，脚本会自动给输出目录加参数后缀，分别保存消融结果。`RUN_TAG` 可自定义输出目录标识。修改 YAML 但未提供环境变量时，请用 RUN_TAG 区分实验。

直接调用 main.py 时，参数是 `--time-layers`、`--variable-layers`、`--cross-layers` 和 `--within-variables both|exogenous|endogenous|none`。保存的 config.json 记录实际配置，evaluation_protocol.json 的 model_parameters 记录新模块设置。

组内关系仍按实际时间前缀计算。实现按接收变量分组，不显式分配全图稠密邻接矩阵。四记录型数据集会在 CPU 整理批次时打包新关系。若外部调用者提供旧版 packed plan，模型会补建组内关系。

新增组内边会增加注意力计算，DIST 等密集数据的训练速度和峰值显存需要在目标 GPU 上实测。
