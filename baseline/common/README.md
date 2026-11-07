# 不规则预测公共协议

TimeXer 与 CrossLinear 共用此处入口、训练器和网格适配契约；现有 DAG 入口及实现未修改。
这两个入口仅从 `config/ExEn/processed/*.yaml` 读取数据路径、内外生变量划分、物理历史/预测
时长、拆分比例及已知未来 X 列表，不采用 ExEn 的模型与训练超参数。

## 数据和时间

复用项目 `ExEnProcessedProcessor`、训练窗口拟合的 `ExEnProcessedMinMaxScaler`、
独立训练 StandardScaler 评价及 DAG 的 float64 时间数据集。
缺失规则、窗口划分、训练统计范围与已有 DAG 基线一致；原始零值为缺失，归一化后的零仍有效。
输入归档为 `time/values/mask/names`，默认路径来自 YAML，可用 `--data` 覆盖。

公共网格层复用 `IrregularDAG.grids/forward`，不实例化也不训练 DAG 网络。
历史长度 H、预测长度 F、历史网格点 N，`dt=H/N`，历史网格从窗起点到预测起点前一格，
预测网格从预测起点到 `ceil(F/dt)*dt`，包含末端支撑点。
输入按真实时间逐变量线性插值；历史 Y/X 严格不读未来，未来 X 仅使用本窗历史与声明已知的
未来 X。边界保持最近端点，单点常值保持，无观测用训练 MinMax 空间的 0 回退，不跨窗取值。
`known_future_exogenous_var_ids: all` 必须表示实际预测时确实已知的 X；可指定列表或 `[]`。

规则输出通过可微分线性插值映射到原始 Y 查询时间；只在真实有效 Y 查询上训练和评价。
padding、未知未来 X、未来 Y 值均不能作输入；原始绝对查询时间直接保存在预测导出中。
多目标采用 YAML target_names 的顺序；最终导出使用原归档变量 ID。

## 训练和评价

两个方法分别使用各自声明的预测目标，无 DAG 辅助损失。
复用仓库 checkpoint/早停生命周期；验证模型选择使用输入 MinMax 单位的查询损失，
先对 batch 中有效查询求均值，再对 batch 等权平均，与上游验证聚合方式一致。
TimeXer 默认 Huber(delta=.5)，CrossLinear 默认 MSE；可用 `--loss` 显式指定。
默认 Adam lr=.0001，可用 `--lr` 调整。

最终六指标 MSE/MAE/RMSE 与对应 `_norm` 指标使用既有仓库协议：先对每个窗口内有效
真实 Y 查询统计，再对窗口等权平均；`_norm` 在独立 train-StandardScaler 空间，其他为原始单位。
因此 checkpoint 选择目标与最终主报告指标 `mse_norm` 有区别，均在 metrics.json 中记录。

输出为最佳 `model.pth`、checkpoint、epoch 记录、`input_scaler.npz`、
`evaluation_scaler.npz`、`metrics.json`、`predictions.npz`。
预测 NPZ 只含有效 Y 查询：`prediction/actual/variable_id/window_id/timestamp`。
metrics 包含配置、模型变体、来源、网格、参数量、优化器和训练/评价协议。
输出目录必须为空，每次实验使用新的目录。不支持此入口的分布式和混合精度。

## 批量运行

在项目根目录运行：

```bash
python -m pip install -r baseline/timexer_irregular/requirements.txt
DEVICE=cuda:0 BATCH_SIZE=16 EPOCHS=100 bash baseline/run_full_exo.sh
```

脚本只运行 FullExo，遍历现有全部 13 份配置（包含 DIST），共 26 次训练；正常模型与完整数据，
每次最多 100 epoch，早停开启。不传 `--tiny/--max-windows`。
可用 `RUN_ROOT` 指定新输出目录、`PYTHON` 指定当前环境解释器。
每个方法/数据集单独日志，失败立即停止。默认仅使用可见的 `cuda:0`，串行运行。

单次入口还支持 `--seq-len`、`--seed`、`--threads`、`--patience`，可通过 `--help` 查看。
`--tiny` 改变网络容量，`--max-windows` 同时限制各拆分和训练 scaler 拟合窗口，仅用于冒烟测试。

## 验证

```bash
python -m pytest tests baseline -q
```

包括原有测试、DAG 测试，以及新增模型的单/多目标、FullExo/NoExo、未来 Y 污染、
未来 X 的接入和屏蔽、空/单点回退、梯度、checkpoint、指标、精确时间戳导出、
上游哈希及全量批处理脚本覆盖测试。CPU 冒烟结果不能替代 CUDA 全量精度实验。
