# DIST：三个模型与统一评估

支持 ExEnGraphTransformer、TPatchGNNFusion、HiPatchFusion。输入 `data/processed/DIST.npz`，不改写原始数据。

## 运行

在仓库根目录、已安装 CUDA 版 PyTorch 的环境中执行：

```bash
python -m pip install -r baseline/patch_fusion/requirements.txt

# 顺序运行三个模型，提供历史和未来 X
DEVICE=cuda:0 bash scripts/run_dist.sh

# 分别运行
MODEL=exen DEVICE=cuda:0 bash scripts/run_dist.sh
MODEL=tpatchgnn DEVICE=cuda:0 bash scripts/run_dist.sh
MODEL=hipatch DEVICE=cuda:0 bash scripts/run_dist.sh

# 历史外生 / 无外生；可与 MODEL 组合
MODE=HistExo DEVICE=cuda:0 bash scripts/run_dist.sh
MODE=NoExo DEVICE=cuda:0 bash scripts/run_dist.sh

# 原 Fusion 脚本也支持 DIST，大小写均可
BACKBONE=tpatchgnn DEVICE=cuda:0 bash scripts/run_patch_fusion.sh DIST
BACKBONE=hipatch DEVICE=cuda:0 bash scripts/run_patch_fusion.sh DIST
```

`run_dist.sh` 支持 MODEL、MODE、DEVICE、PYTHON、RUN_TAG。默认输出按模式和模型区分：
`results/dist_variable_macro_FullExo/<模型>/DIST/`，模型文件对应 `model_states/`。
ExEnGraphTransformer 可设置 `ATTENTION_DTYPE=auto`，默认 FP32。

原直接启动命令现在也采用新评估：

```bash
python -u main.py --config_path config/ExEnGraphTransformer/processed/DIST.yaml \
  --device cuda:0 --model_save_dir_path model_states/exen_graph_DIST_FullExo \
  --result_save_dir_path results/exen_graph_DIST_FullExo
```

两个 Fusion 的配置分别是 `config/TPatchGNNFusion/processed/DIST.yaml` 和
`config/HiPatchFusion/processed/DIST.yaml`。直接启动也支持 `--history-only-exogenous` 或 `--no-exogenous`。
请使用新的输出目录，不复用旧 DAG 评估实验的结果或 checkpoint。

## 数据任务

- 保留历史 7200 秒、预测 1800 秒；按历史加预测总长构造不重叠窗口，然后按时间 60%/20%/20% 划分，过滤无有效历史或未来目标的窗口。
- Y 为 FIC05、FIC09、FIC11、LIC02、LIC03、LIC04、LIC05、LIC06、LIC12 的 `.PV`；X 为对应 9 个 `.OP`。真实顺序来自文件 `names`。
- 沿用 DIST 原规则：原 mask 有效、数值有限且原始值非零才有效；MinMax 后产生的零仍有效。
- 三模型 batch=1，MinMax 仅用有效训练窗口观测拟合，不使用验证和测试。验证/测试可能超出 [0,1]，不裁剪。
- FullExo 提供未来真实 OP，是给定条件实验，不自动证明 OP 在部署时提前可知或因果外生。

## 新评估

三个配置使用 `TPatchForecastTrainer`，在模型输入 MinMax 空间内：跨整个验证/测试集累计各目标变量的平方误差、绝对误差和有效数，再对有观测的目标变量等权平均。

```text
MSE = mean_v(sum_valid(error_v²) / count_v)
MAE = mean_v(sum_valid(abs(error_v)) / count_v)
RMSE = sqrt(MSE)
```

训练损失为每批变量宏平均 MSE；验证集 `mse` 用于早停和选择 checkpoint。
不使用额外 DAG StandardScaler、`mse_norm` 或逐窗口指标等权平均。
聚合方式与四个 tpatch 数据集一致，但 DIST 仍仅在训练窗口拟合缩放，不照搬临床/Activity 的 train+validation seen 拟合规则。

`metrics.json`、`evaluation_protocol.json` 记录指标空间、缩放、变量、模式、模型参数和窗口长度。
`test_y_pred.npy`、`test_y_true.npy` 在 MinMax 空间，配套 `test_y_mask.npy`、`test_var_idx.npy`、`test_window_id.npy`。
`test_window_start_seconds.npy` 给出每个窗口起点；DIST 无患者/记录 ID，不生成 `test_record_id.npy`。

## Fusion 约定

历史 Y/X 按 60 秒打包，2 小时共 120 个 patch；每个有效历史观测均保留，不进行规则重采样、平均或截断。
时间使用相对窗口的秒数除以总长 9000 秒。
原骨干在实际未来 Y 查询时刻产生预测，提取 Y 向量，与同一时刻的 X 通过 DAG 两层 ReLU MLP（隐藏宽度 64、无残差）融合。
X 使用不晚于查询的最近有效值，可回退至历史；均无值时填 0。不使用查询之后的 X 反向填充，未来 Y 仅为监督。
HistExo 将融合端未来 X 置 0；NoExo 同时屏蔽历史 X；保留相同变量槽和参数规模。

DIST 没有两个骨干的官方数据集配置。当前为运行起点：hidden=32、1 层、1 头、lr=0.001、30 轮/5 轮早停；60 秒 patch 用于控制密集记录的图大小，不宣称最优。
ExEnGraphTransformer 保留原 hidden=32、4+1+3 层（时间/组内变量/X→Y）、lr=0.003 和 attention/checkpoint 行为。
Hi-Patch 图内边随 patch 观测数平方增长，实际显存和吞吐需在目标 GPU 测量；batch=1 不保证所有窗口均适合任意 GPU。

```bash
python -m pytest -q tests/test_dist_fusion.py tests/test_patch_fusion.py tests/test_tpatch_packed.py
```
