# DAG 不规则时序对比基线

本目录是独立实验入口，不修改 `main.py`、现有模型、训练器或数据协议。基于
`decisionintelligence/DAG@0758990e2c73bb54138ea3e7b11a35cbc5476bcc`；
`vendor/PROVENANCE.json` 记录来源与原文件 SHA256，附 MIT LICENSE。
TC、CC、融合 `alpha*TC+(1-alpha)*CC`、预测头及归一化计算均保持原样，仅调整包内导入路径及一个文件末尾空行。

## 运行

在仓库根目录安装与运行（数据不随代码提交）：

```bash
python -m pip install -r baseline/dag_irregular/requirements.txt
python -m baseline.dag_irregular.run \
  --config config/ExEn/processed/DIST.yaml \
  --mode FullExo --output baseline/dag_irregular/runs/DIST_FullExo
python -m baseline.dag_irregular.run \
  --config config/ExEn/processed/DIST.yaml \
  --mode NoExo --output baseline/dag_irregular/runs/DIST_NoExo
```

可使用现有 `config/ExEn/processed/*.yaml` 的任意数据集配置；读取其中的数据路径、目标变量、
历史/预测物理时长、拆分比例和已知未来外生变量列表。`--data /path/to/file.npz` 覆盖路径。
只消费这些配置的数据部分，**不继承 ExEn 模型或优化器超参数**。
DAG 参数可在 YAML 顶层 `dag:` 下指定，例如：

```yaml
dag:
  seq_len: 96
  patch_len: 16
  stride: 8
  d_model: 512
  d_ff: 2048
  n_heads: 8
  e_layers: 2
  dropout: 0.1
  alpha: 0.2
  beta: 0.1
```

默认 CPU，可指定 `--device cuda:0`；默认 100 epoch、batch size 256、Adam lr=.02、
DAG type3 学习率调整、验证 MAE 早停 patience=10。资源不足可调整 batch size。
小规模执行加 `--tiny --epochs 2 --max-windows 2 --batch-size 2`。
`--tiny` 会改变模型容量，`--max-windows` 会限制各拆分及训练统计的拟合窗口，均只适合冒烟测试。

## 数据与时间语义

复用 `ExEnProcessedProcessor` 的物理时间不重叠窗口、按时间顺序 train/valid/test 拆分、
变量角色以及缺失规则：`original_mask & finite(raw_value) & (raw_value != 0)`。
输入 `time` 必须有限、严格递增；`values/mask/names` 的协议与现有 processed NPZ 相同。
标准化复用训练窗口拟合的 `ExEnProcessedMinMaxScaler`；评价独立复用训练窗口拟合的
`DAGEvaluationScaler`。验证/测试不拟合统计量；训练集未来标签用于训练统计是既有协议。

`DAGProcessedDataset` 复用现有角色划分、padding 和 collation，并保留 float64 相对时间与
原始绝对时间。有效 Y 查询始终来自原始观测，绝不补造监督标签。原始绝对时间直接导出。

上游 DAG 没有时间戳输入，`DAGModel` 返回 `[B,pred_len,target_dim]`。
上游滑窗将历史后紧邻的第一步作为输出索引 0。本适配器以预测起点 `o` 为该时间位置：
历史长度 H、预测长度 F、历史网格点数 N，令 `dt=H/N`。
历史网格为 `o-H + k*dt (k=0..N-1)`；预测网格为
`o + k*dt (k=0..ceil(F/dt))`。
因此 `pred_len=ceil(F/dt)+1` 自动派生，不能另行传入。末端额外支撑点覆盖所有 `[o,o+F)`
查询，避免把末段查询挤到同一个预测步。这是显式时间契约，末端支撑点没有虚构 Y 标签。

每个变量分别按真实时间线性插值：

| 输入/边界 | 策略 |
|---|---|
| 历史 Y、历史 X | 只读取 `[o-H,o)` 的相应角色；未来 X 也不能填历史网格 |
| 未来 X | 读取本窗历史 X 与明确声明已知的 `[o,o+F)` 未来 X；未声明的未来 X 在数据层移除 |
| 两观测之间 | 线性插值；整个已知未来 X 可参与插值，前提是它们确实在预测时已知 |
| 左/右边界 | 最近端点常值保持，不线性外推、不跨窗口取值 |
| 单观测 | 常值保持 |
| 整窗无可用值 | 归一化空间常数 0（训练 MinMax 的最小值），不从标签或其他拆分回填 |
| 重复时间 | 插值函数按时间聚合均值；processed 输入本身要求时间严格递增 |
| 无效值/padding | 不作插值源；查询时间越界或非有限则报错 |

DAG 的规则输出通过可微线性插值映射到**原始有效未来 Y 时间**，梯度回传 TC、CC 及预测头。
非查询输出槽仅为兼容仓库接口的零占位，不参与损失、评价或结果导出。

## FullExo / NoExo

FullExo 输入历史 Y、历史 X、声明已知的未来 X；`known_future_exogenous_var_ids: all`
必须意味着这些变量在实际预测时确实可获得。可配置列表或 `[]`；未提供未来观测时，
只用历史 X 作保持回退，不能使用未声明的未来观测。

NoExo 在数据层删除所有历史与未来 X，在适配器再次忽略全部 X，关闭 DAG 的
`use_t_exog/use_c_exog`。上游 TC/CC 内部仍会运行外生投影，并不能直接处理零宽通道，
所以仅构造**固定零值的一维兼容通道**供原计算路径使用。此通道不含任何实际外生信息；
TC/CC、融合及预测头仍保留。两项外生辅助损失在该模式为零。
这是 Y-only 的输入消融，兼容通道会使参数量与 FullExo 不同；报告会记录参数量。
NoExo 仍读取同一份 processed 归档和训练统计，以保持样本与评价协议一致。

## 训练、评价与产物

复用仓库 `ExEnDAGForecastTrainer` 的评价和 `AbstractTrainer` 的 checkpoint/早停生命周期。
实验子类恢复 DAG 默认训练目标：有效真实查询 MAE +
`beta*(TC未来X重构MAE + CC历史Y重构MAE)`。
辅助项保持上游对规则网格的原始计算，因此可能包含插值或回退的输入；
主预测损失与所有评价指标仅用真实有效 Y 查询。

采用 Adam、type3 学习率日程；与上游一样，验证选择使用不含辅助损失的 MAE（输入单位），
训练按 batch 中有效查询求均值；验证 MAE 等权平均 batch。与已有 ExEn 的 `mse_norm`
checkpoint 选择不同，明确记录为 `query_mae_input_scaled`。
六项评价仍为既有协议：原始单位及 train-StandardScaler 单位的 MSE/MAE/RMSE，
窗内只计有效 Y 查询，再对窗口等权平均。没有用插值标签评价，也不是全点微平均。
不支持本入口的分布式或混合精度；使用 FP32 模型和 float64 时间。

输出目录包含最佳 `model.pth`、仓库 checkpoint/epoch 记录、两种训练 scaler、
`metrics.json`（配置、拆分、网格、指标、参数量等）与 `predictions.npz`。
NPZ 仅包含有效 Y 查询的 `prediction/actual/variable_id/window_id/timestamp`，预测和标签为物理单位。
每次实验请指定独立输出目录。

## 验证

```bash
python -m pytest baseline/dag_irregular -q
python -m pytest tests -q
```

新增测试覆盖边界/缺失/单点/重复时间、插值梯度、TC/CC 反传、未来 Y 污染、历史与未来 X 隔离、
NoExo 对 X 值/时间/缺失掩码的不变性、末端时间位置、越界查询、上游源码哈希、两模式的完整
训练/最佳模型重载/评价/导出、原始查询时间精确保留。
实际验证摘要见 `validation.json`；DIST 两模式只进行了小规模冒烟训练，不代表模型性能比较。
