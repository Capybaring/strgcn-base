# 直接使用 processed 数据

评估已切换为 DAG 的训练集 StandardScaler 空间，主指标是 `mse_norm`。模型输入仍用 MinMax；两者独立。详见 [评估量纲与对比规则](DAG_EVALUATION_zh.md)。

## 数据放在哪里

将你提供的整个 `processed` 文件夹复制到**仓库根目录的 `data/` 下**，最终目录是：

```text
strgcn-base/
├── main.py
├── environment-exen.yml
├── config/ExEn/processed/
└── data/
    └── processed/
        ├── BE.npz
        ├── NP.npz
        ├── DIST.npz
        ├── BE_irregular.csv
        ├── NP_irregular.csv
        └── table3_irregular/
            ├── PJM.npz
            ├── FR.npz
            ├── DE.npz
            ├── Energy.npz
            ├── Sdwpfm1.npz
            ├── Sdwpfm2.npz
            ├── Sdwpfh1.npz
            ├── Sdwpfh2.npz
            ├── Colbun.npz
            └── Rapel.npz
```

程序直接读取 NPZ；CSV 和 manifest 可保留用于核对。无须重新导出长表，也无须插值或再次随机删除观测。
`data/` 已被 Git 忽略，数据不会随代码推送。原始文件不被写回。

## conda 和运行命令

从仓库根目录执行：

```bash
conda env create -f environment-exen.yml
conda activate strgcn-exen
python -m pytest -q tests

# BE
python main.py \
  --config_path config/ExEn/processed/BE.yaml \
  --model_save_dir_path model_states/exen_BE \
  --result_save_dir_path results/exen_BE

# DIST：历史 2 小时，预测 30 分钟
python main.py \
  --config_path config/ExEn/processed/DIST.yaml \
  --model_save_dir_path model_states/exen_DIST \
  --result_save_dir_path results/exen_DIST
```

已建立环境时，也可 `conda activate strgcn-exen` 后运行 `python -m pip install -r requirements-exen.txt`。
默认 CPU；使用已安装对应 PyTorch 的 GPU 环境时，把 YAML 的 `train_config.device` 改成 `cuda:0`。
CPU 及数据读取、反向传播已验证；尚未验证 CUDA 或混合精度。

其他数据集使用同目录下同名 YAML，输出目录也相应更名。
如只想检查入口，可临时在 `data_processor_params` 中添加 `max_windows_per_split: 4`，并将 `max_epoch_num` 设为 1。
限制窗口仅用于检查；正式比较应删除限制。某些数据集的小样本训练段可能没有某个变量的非零观测，此时程序明确报错，不用评估数据补统计。

## 0 值和 mask 的规则

按本次用户要求，针对这些 processed 文件：

```python
effective_mask = original_mask & isfinite(raw_values) & (raw_values != 0)
```

**所有原始 0 都作为缺失值**，即便原 NPZ 的 mask 为 True。被排除的位置不生成输入观测节点或监督标签。
因此旧 manifest 中的观测数量和旧指标不再直接代表本次实验，尤其降水、风电功率等包含较多 0 的变量。
NaN/Inf 也视为缺失；已有 mask=False 的位置始终缺失。
原始非零观测经过 MinMax 缩放后可能恰好等于 0，这仍然是有效观测，不进行第二轮“零值删除”。
未来查询节点在模型内部使用 unknown token；其输入占位值为 0 不意味着删除查询节点。

## 数据切分、窗口和缩放

直接读取 `time / values / mask / names`，时间单位是秒；变量 ID 是 `names` 中的列索引，保留顺序和名称中的空格。

- 从第一个时间戳开始，构造时长为“历史+预测”的完整、不重叠窗口。
- 按候选窗口的时间顺序分配 60%/20%/20% 为训练/验证/测试，再跳过无历史或无有效未来目标的窗口。三个分区不共享窗口时间。
- 丢弃最后不完整窗口；不推测最后时间戳之后的数据覆盖范围。
- 查询目标只在有效、非零的未来内生位置计算损失；历史和未来外生值按可用性处理。
- MinMax 统计仅使用训练窗口的非零有效观测，包括训练窗口的监督值；验证和测试不参与拟合。
- 按窗口生成长表，归一化为窗口内相对时间，不对整条序列展开后再复制多份。

窗口配置位于 `data_config.data_processor_params`，不需要修改模型：

| 配置文件 | 历史 | 预测 | 内生目标 |
|---|---:|---:|---|
| BE、NP、PJM、FR、DE | 168 小时 | 24 小时 | OT |
| Energy | 168 小时 | 24 小时 | Thermoelectric |
| Sdwpfm1、Sdwpfm2 | 84 小时（168 个半小时） | 12 小时（24 个半小时） | Patv |
| Sdwpfh1、Sdwpfh2 | 168 小时 | 24 小时 | Patv |
| Colbun、Rapel | 60 天 | 10 天 | Water_Level_daily_avg |
| DIST | 2 小时 | 30 分钟 | 9 个 PV |

各配置沿用此前参考实验的短窗口时长。不是滑动一步预测协议；如需与其他实验比较，须统一窗口、切分、零值规则和误差空间。
YAML 中时长可以写秒数或 `168h`、`60d` 等带单位字符串。

## 已知未来外生信息

各配置默认：

```yaml
dataset_params:
  known_future_exogenous_var_ids: []
```

即默认只使用历史外生，不把存档中的事后外生观测自动当作已知未来输入。
如果 BE/NP/PJM/FR/DE 的两个预测量在你的任务中确实已提前获得，可改成 `[1, 2]`。
Energy 为 `[1,2,3,4,5]`，Sdwpf 为 `[1,2,3,4,5,6]`，Colbun/Rapel 为 `[1,2]`，DIST 的 OP 为 `[9,10,11,12,13,14,15,16,17]`。
只填写预测发布时可用的那部分；空列表和任意合法子集均支持。
未配置的未来外生节点在生成样本时即被移除，减小图规模。

## 内存、输出和测试

DIST 已默认启用 [CSR 稀疏关系 backend](SPARSE_RELATIONS_zh.md)，只计算合法边。其他配置继续使用原 dense 分块模式。

`receiver_chunk_size: 128` 按接收节点分块构建掩码、关系和聚合，训练时重算块内中间结果以降低内存。
没有新增超节点或改变候选边；测试核对了与整图计算的输出和参数梯度一致。
计算总量仍为平方级；DIST 默认 batch_size=1，完整训练可能明显慢于原 Sandwich。

结果目录除原有 `result.json`、`test_y_pred.npy` 和 `test_y_true.npy`，还保存
`test_y_mask.npy` 与 `test_var_idx.npy`。后处理时使用 mask=True 的位置计算目标误差，数组位置逐一对应。
默认主指标 `mse_norm/mae_norm/rmse_norm` 在独立训练集 StandardScaler 空间计算，另报告原单位的 `mse/mae/rmse`。预测/真值数组保存原单位；`do_inverse_scaling` 保持 false，由新训练器负责转换，避免重复缩放。结果另含 `metrics.json`、`evaluation_protocol.json` 和窗口 ID。

28 项测试通过，验证内容包括零值/原 mask 合并、数据源不变、时间分区无重叠、原始零与缩放零的区别、训练集缩放统计及保存加载、
已知未来开关、目标掩码、分块输出/梯度一致性和完整 CLI。
13 个实际数据集已在本次零值规则下完成完整窗口切分、训练统计拟合及单窗口前向/反向检查；DIST 使用实际 2h→30min 窗口。
BE 另作每分区 4 个窗口、1 epoch 的训练/验证/测试入口检查。这些检查不构成完整训练后的精度结论。
