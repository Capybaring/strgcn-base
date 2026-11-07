# TimeXer 不规则预测基线

本目录提供 **TimeXer-DAG-future-encoder**：采用 DAG 基准仓库中已增加未来 X 编码的 TimeXer。
该版本保留内生 patch/global token、自注意力、跨变量注意力、融合和预测头，支持多个内生目标。
它与原版 TimeXer 历史外生输入版本有区别：将历史 X 与已知未来 X 沿时间拼接后编码，
外生编码层长度为 `seq_len + pred_len`，不是额外接在输出上的 MLP。

来源为 `decisionintelligence/DAG@0758990e2c73bb54138ea3e7b11a35cbc5476bcc` 的
`ts_benchmark/baselines/timexer/`；仅重定位导入路径，具体文件和原始 SHA256 见
`vendor/PROVENANCE.json`，附该参考仓库 MIT LICENSE。原始方法仓库为
https://github.com/thuml/TimeXer 。

在项目根目录安装与运行：

```bash
python -m pip install -r baseline/timexer_irregular/requirements.txt
python -m baseline.timexer_irregular.run \
  --config config/ExEn/processed/DIST.yaml \
  --mode FullExo --device cuda:0 --epochs 100 --batch-size 16 \
  --output results/baseline/TimeXer/DIST_FullExo
```

运行全部 13 个数据集及 CrossLinear：

```bash
DEVICE=cuda:0 BATCH_SIZE=16 EPOCHS=100 bash baseline/run_full_exo.sh
```

共享的数据处理、时间语义、训练与评价说明见 [公共协议](../common/README.md)。
以现有数据配置为基础，可在 YAML 顶层增加 `timexer:` 配置模型参数：

```yaml
timexer:
  seq_len: 96
  patch_len: 16
  d_model: 512
  d_ff: 2048
  n_heads: 8
  e_layers: 2
  dropout: 0.2
  factor: 1
  activation: gelu
```

默认训练目标为 DAG 基准 TimeXer 的 Huber（delta=.5），Adam lr=.0001、type3 学习率日程；
这与原始 TimeXer 仓库的 MSE 设置有区别，可通过 `--loss mse` 显式切换。
模型选择使用相同目标的验证查询损失，最终仍报告仓库统一的六个指标。
最大 epoch 默认 100、早停 patience 默认 10、batch 默认 16，均可通过命令行覆盖。

`--mode NoExo` 删除真实 X，使用固定零兼容通道，关闭未来 X 编码输入；仍保留原网络结构。
小规模验证加 `--tiny --epochs 2 --max-windows 2 --batch-size 2`。
输出目录必须为空，避免覆盖已有模型。验证摘要见 `validation.json`；只验证运行流程，不作精度结论。
