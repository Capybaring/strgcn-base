# 与 DAG 对齐的评估量纲

## 当前主指标

processed 配置使用 `ExEnDAGForecastTrainer`，早停与最佳模型选择依据 **`mse_norm`**。

| 指标 | 空间 | 定义 |
|---|---|---|
| mse_norm、mae_norm、rmse_norm | 训练集 StandardScaler 标准化空间 | 原单位预测和真值分别按训练均值/标准差转换后计算 |
| mse、mae、rmse | 原单位空间 | 还原模型输入缩放后计算 |
| train_loss | 模型输入的训练集 MinMax 空间 | 保留原 ExEn 训练 MSE；不作为最终对比指标 |
| validation/test loss | StandardScaler 空间 | 等于 mse_norm，用于早停 |

训练输入缩放与评估缩放是两个独立对象。模型仍可使用 MinMax 输入：

```text
模型输出（MinMax）
  → 逆变换到原单位：prediction_raw = prediction_model * train_range + train_min
  → DAG 评估标准化：z = (value_raw - train_mean) / train_std
  → 计算 mse_norm / mae_norm / rmse_norm
```

均值、标准差使用原始训练窗口中有效且非零的观测拟合，标准差采用 sklearn StandardScaler 的总体标准差（ddof=0）；常量通道的 scale=1。
不将 MinMax 结果再次当成原始数值直接拟合评估 scaler，不使用验证/测试数据拟合统计。
0 缺失判定只用于原始数据。标准化后的真值恰好为 0 仍是有效标签。

## 窗口汇总

DAG 的 rolling_forecast 先逐个预测窗口调用指标，再对窗口结果取算术平均。本实现复用该汇总语义：

1. 每个预测窗口仅统计有效未来内生目标；窗口内对所有有效目标单元取平均。
2. 不同窗口等权，即使观测数量不同；不按 batch 平均，不按有效点数给窗口加权。
3. `rmse_norm = mean_window(sqrt(mse_norm_window))`。通常不等于 `sqrt(mean_window(mse_norm_window))`。

原始 DAG 针对规则完整矩阵，没有不规则 mask。本实现的缺失数据扩展是“只选择有效单元，再使用同一窗口指标公式”；不把缺失位置当成 0 参与分母。
不再把 exo-base 旧版的“全测试集按变量累加后宏平均”和 signed-MAPE 当成 DAG 的默认指标。
本次只报告六项 MSE/MAE/RMSE，不以未定义零分母处理方式替代 DAG 的其他百分比指标。

## 输出文件与复核

训练命令及 `data/processed/` 位置不变。请使用新的输出目录，避免混用旧指标或旧 checkpoint：

```bash
python main.py \
  --config_path config/ExEn/processed/BE.yaml \
  --model_save_dir_path model_states/exen_BE_dag_eval \
  --result_save_dir_path results/exen_BE_dag_eval
```

模型目录增加 `evaluation_scaler.npz`，保存评估均值、scale、有效数量及变量顺序；原 `scaler.npz` 继续保存模型输入的 MinMax 参数。

结果目录：

- `metrics.json`：主指标、原单位指标、各目标有效数量及验证历史。
- `evaluation_protocol.json`：指标空间、训练统计、汇总方式、缺失规则和未来外生开关。
- `result.json`：兼容原训练器的结果结构，test_result 同样包含新指标。
- `test_y_pred.npy`、`test_y_true.npy`：**原单位**数组，不是 MinMax 或标准化数组。
- `test_y_mask.npy`、`test_var_idx.npy`：与上述数组逐位置对应；只选 mask=True 的目标位置。
- `test_window_id.npy`：每个数组位置所属的测试窗口序号。
- `test_window_start_seconds.npy`：按窗口序号索引的原始时间起点；验证和测试不打乱顺序。

不要把整个预测数组直接取均值作为 mse_norm：应先按目标 ID 使用评估 scale 转换误差，再按窗口分组汇总。
`do_inverse_scaling` 保持 false；新训练器自行管理“还原原单位→评估标准化”的完整转换，开启该旧开关会明确报错，避免重复缩放。
旧预切分 demo 仍使用原训练器；本次默认切换的是 `config/ExEn/processed/` 下的全部 13 个真实数据配置。

## 参考证据与可比性范围

已读取并固定参考版本：

- [DAG @0758990：评估 scaler](https://github.com/decisionintelligence/DAG/blob/0758990/ts_benchmark/evaluation/strategy/forecasting.py)：`_get_eval_scaler` 在训练段拟合 StandardScaler，独立于模型训练缩放。
- [DAG：指标函数](https://github.com/decisionintelligence/DAG/blob/0758990/ts_benchmark/evaluation/metrics/regression_metrics.py)：原单位 `mse/mae/rmse` 和 `*_norm` 的转换公式。
- [DAG：滚动结果汇总](https://github.com/decisionintelligence/DAG/blob/0758990/ts_benchmark/evaluation/strategy/rolling_forecast.py)：逐窗口计算指标后对窗口取平均。
- [exo-base @2fa42e0：当前指标实现](https://github.com/Capybaring/exo-base-model/blob/2fa42e0/exobase/metrics.py)：目前仍在 MinMax 空间累计误差，再对变量宏平均。这与官方 DAG 的量纲/窗口汇总不同。

直接调用官方 DAG 的六个指标函数，与此实现用相同训练统计、相同完整窗口输入逐项比对已通过，见 `reports/dag_metric_parity.json`。
单元测试另行覆盖不等观测数窗口、不同目标尺度、归一化后零值、批次大小不变性、原单位导出和结果重算。
BE 使用每分区 4 个窗口、1 epoch 跑通新评估流程，未进行全量精度实验。

本次修改 STRGCN 仓库，不会自动改变 exo-base 旧结果的量纲。exo-base 需要用同一 DAG 规则重新评分后才能与新指标比较，不能仅将旧字段改名为 mse_norm。
此外，当前 processed 入口仍是非重叠窗口和有效观测 mask，未替换为官方 DAG 的规则网格 stride=1 滚动协议。
当前 DIST 配置是 2h→30min，而所读 exo-base 仓库的 DIST 是 4h→2h；本次评估修改保留现有预测任务。
模型对比还需统一训练/测试窗口、预测跨度、0 缺失规则、未来外生可用性及训练统计。指标公式与量纲一致不代表完整实验协议已一致。
