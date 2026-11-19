# DIST：三个基座与五种方法

支持 t-PatchGNN、Hi-Patch、mTAND 的 original/base/mlp/v1/strgcn，共15个组合。
复用当前网络与融合模块，不修改原始 `data/processed/DIST.npz`。
三个配置位于 `config/{TPatchQueryFusion,HiPatchQueryFusion,MTANQueryFusion}/processed/DIST.yaml`。

## 任务协议

- 输入：DIST.npz 的原始 time/values/mask/names；原始零、非有限值和原mask缺失均排除。
- 9个Y：FIC05.PV、FIC09.PV、FIC11.PV、LIC02.PV、LIC03.PV、LIC04.PV、LIC05.PV、LIC06.PV、LIC12.PV。
- 9个X：上述同名的 OP 列。
- 历史7200秒，预测1800秒；按9000秒不重叠窗口、时间顺序60/20/20划分，保持旧DIST配置。
- 只用训练窗口拟合MinMax，验证/测试不参与；评估在此输入空间，逐变量汇总误差后宏平均。
- 五种方法均评估9个Y，使用相同划分和有效Y窗口。
- original：全部18变量历史与训练目标，无未来值输入；只在评估时筛选9个Y。
- base：Y-only历史、Y-only训练；mlp：现有最近可见X填充与两层MLP；
  v1：现有逐点X编码及因果时间掩码；strgcn：STRGCN关系编码X，允许窗口内全部已提供X影响任意Y查询。
- strgcn与v1的未来X可见性不同；要单独比较X编码器，可另跑已有 v1_all 对照。
- 沿用旧DIST patch配置：patch=60秒、隐藏维度32、一层、batch=1。mTAN隐藏维度64，
  latent32、gen_hidden50、参考点64，保持其现有网络默认设置；patch仅作其CPU元数据整理。
- 单seed3407，最多30轮、5轮早停。原先四数据集的设置不变。

## 一次运行15个组合

```bash
bash scripts/run_dist_query_fusion.sh --device cuda:1
```

只运行一个基座的五种方法：

```bash
bash scripts/run_dist_query_fusion.sh hipatch --device cuda:1
bash scripts/run_dist_query_fusion.sh mtand --device cuda:1
bash scripts/run_dist_query_fusion.sh tpatchgnn --device cuda:1
```

只运行一个组合：

```bash
bash scripts/run_query_fusion.sh hipatch dist strgcn --device cuda:1
```

DIST路径通过 `DIST_DATA_PATH` 指定，默认 ./data/processed/DIST.npz。
支持既有 DEVICE/GPU/PYTHON/BATCH_SIZE/RUN_DIR/DRY_RUN 设置；显式 --device 规则保持一致。
每次创建独立结果目录，拒绝覆盖已有结果。

Python矩阵入口通过 `--datasets dist --dist-data /path/to/DIST.npz` 选择；
`--max-samples-per-split` 在DIST映射为每划分最多多少个窗口，不能把诊断结果当全量结果。
DIST不要求存在 tpatch-data；旧 `--data-root` 参数为兼容保留，由shell入口自动提供。

## 验证

2026-11-19：18项DIST测试及152项现有回归测试通过，共170项。
真实DIST数据上，小模型15次、默认模型维度15次CPU单轮训练均成功，每划分最多1个窗口。
30次运行均检查Y标签、评估mask，并独立重算MSE/MAE/RMSE；五种方法的Y标签相同。
详见 `reports/dist_query_validation.json`。不代表已完成全量GPU训练或证明精度提升。
