# 对比实验基线

各方法分别放在独立子文件夹中，复用项目的数据划分、标准化和评价协议。

| 方法 | 目录 | FullExo 的未来 X 接入方式 |
|---|---|---|
| DAG | [dag_irregular](dag_irregular/README.md) | 原 DAG 的 TC/CC |
| TimeXer | [timexer_irregular](timexer_irregular/README.md) | DAG 基准的未来 X 外生编码扩展 |
| CrossLinear | [crosslinear_irregular](crosslinear_irregular/README.md) | 官方原模型 + DAG 式 FutureX-MLP |

三个方法均支持 FullExo 与严格 Y-only NoExo，提供可微不规则网格适配、真实查询点评价。
新增两个方法的 [公共协议](common/README.md) 和各自 `validation.json` 记录具体版本与验证范围。

在项目根目录运行 TimeXer 与 CrossLinear 的全部 13 个数据集，仅 FullExo：

```bash
python -m pip install -r baseline/timexer_irregular/requirements.txt
DEVICE=cuda:0 BATCH_SIZE=16 EPOCHS=100 bash baseline/run_full_exo.sh
```

默认结果目录自动附加时间，DAG 已有入口保持原样。

四个tpatch数据集另提供 [t-PatchGNN / Hi-Patch + DAG MLP Fusion](patch_fusion/README.md)。
它们复用TPatchForecastTrainer及当前变量索引，支持FullExo、HistExo、NoExo；
不使用上面processed数据的DAG评价协议。

```bash
DEVICE=cuda:0 bash scripts/run_patch_fusion.sh
BACKBONE=hipatch DEVICE=cuda:0 bash scripts/run_patch_fusion.sh activity
```
