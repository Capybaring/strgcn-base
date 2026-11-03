# 外生输入消融：完整输入、仅历史外生、完全无外生

三个条件的区别：

| 条件 | use_exogenous | known_future_exogenous_var_ids | 模型输入 |
|---|---|---|---|
| 完整模型（默认） | true | all | 历史 Y、历史 X、已知未来 X |
| 无未来外生 | true | [] | 历史 Y、历史 X |
| 完全无外生 | false | 任意设置均不提供 X | 仅历史 Y |

`use_exogenous: false` 在样本生成阶段移除所有历史和未来外生节点，不靠填 0 代替删除。
有效未来内生标签仍变为 unknown-token 查询，标签值不进入模型。
保持同一个 processed 文件、候选窗口、数据切分和目标变量编号；不会因为关闭 X 而重新切分或重新过滤窗口。
模型输入/评估缩放流程沿用同一训练集、按变量独立拟合，因此 Y 的 MinMax 和 StandardScaler 参数保持一致。
参数结构不变，跨角色模块没有外生消息源；当前模型没有不同内生变量之间的边，所以各目标主要利用自身历史。
若某个固定窗口没有某目标的历史观测，该查询仍保留，避免消融改变评估目标集合。

## 单数据集：命令行开关

无需修改原配置，增加 `--no-exogenous`：

```bash
conda activate strgcn-exen
python -u main.py \
  --config_path config/ExEn/processed/BE.yaml \
  --no-exogenous --device cuda:0 \
  --model_save_dir_path model_states/exen_no_exogenous/BE \
  --result_save_dir_path results/exen_no_exogenous/BE
```

CUDA 不可用时将 `--device cuda:0` 改为 `--device cpu`。CLI 开关仅支持 ExEnProcessedDataset。
也可在 YAML 的 `data_config.dataset_params` 中添加 `use_exogenous: false`，省略时默认 true。
CLI 开关优先设置 false；完整模型请移除命令行开关，并确保 YAML 没有显式关闭 X。

## 批量运行：不含 DIST

```bash
# 完全无外生
DEVICE=cuda:0 bash scripts/run_exen_no_exogenous_except_dist.sh

# 完整输入，对照
DEVICE=cuda:0 bash scripts/run_exen_future_except_dist.sh
```

无外生脚本运行 12 个数据集，输出到 `model_states/exen_no_exogenous/<数据集>/` 与
`results/exen_no_exogenous/<数据集>/`；完整模型输出到 `exen_known_future` 对应目录。
不指定 DEVICE 则按配置的设备运行。用 RUN_TAG 设置新的实验目录，避免覆盖先前结果。

## 如何核对

`evaluation_protocol.json` 与 `metrics.json` 中的 protocol 会记录：

```json
{
  "use_exogenous": false,
  "forecast_inputs": "Y_history_only",
  "known_future_exogenous_var_ids": []
}
```

完整输入记录 `Y_history_X_history_and_known_future`，仅历史外生记录 `Y_history_and_X_history`。
按相同目标掩码、窗口起点、缩放统计、预测跨度和训练预算比较 mse_norm / mae_norm；分别训练各模式。
此前仅有 known_future_exogenous_var_ids=[] 的结果属于“无未来外生”，不是“完全无外生”。

## 结论边界

该实验测量外生输入的预测贡献，不识别干预因果效应。当前连接方向来自先验约束，参数通过预测损失学习；
CSR 存储中的稀疏性也来自预设合法边，不是从数据识别出的因果图。
不需要为了可以消融而专门新增模块；要主张因果结构学习，需要另外明确识别假设、结构/滞后学习机制及具有因果真值或干预证据的验证。
