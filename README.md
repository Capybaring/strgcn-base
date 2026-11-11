
# STRGCN
The code for **STRGCN: Capturing Asynchronous Spatio-Temporal Dependencies in Irregular Multivariate Time Series**


## Model Overview
<p align="center">
  <img src="./figs/main.png" alt="STRGCN overview" width="95%">
</p>

## Main Exp
<p align="center">
  <img src="./figs/STRGCN_exp.png" alt="Main Exp" width="95%">
</p>
Comprehensive experiments on four forecasting and four classification benchmarks demonstrate that STRGCN achieves state-of-the-art accuracy with competitive computational and memory efficiency.


## Environment

```bash
conda create -n strgcn python=3.12 -y
conda activate strgcn

# Example: CUDA 12.5 wheels (pick the one that matches your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu125

# Other dependencies
pip install -r requirements.txt
```

## Datasets & Preprocessing

To keep things consistent and reproducible, **we use the same data preparation pipeline as Hi-Patch**.
Please follow their instructions to download and preprocess datasets:

* Hi-Patch repository: [https://github.com/easonLuo2001/Hi-Patch](https://github.com/easonLuo2001/Hi-Patch)


> **Note:** Any licenses/credentials required by the original datasets (e.g., MIMIC-III ) still apply. Obtain access through the official channels before running Hi-Patch preprocessing.

## Quick Start


### Forecasting

```bash
bash scripts/STRGCN_fcst_Physionet2012.sh
bash scripts/STRGCN_fcst_MIMIC3.sh
bash scripts/STRGCN_fcst_Activity.sh
bash scripts/STRGCN_fcst_USHCN.sh
```

### Classification

```bash
bash scripts/STRGCN_cls_Physionet.sh
bash scripts/STRGCN_cls_MIMIC3.sh
bash scripts/STRGCN_cls_P12.sh
bash scripts/STRGCN_cls_P19.sh
```

## Ex-En baseline (single observation graph)

The new `ExEn` model retains STRGCN's value/time encoders, source low-rank
convolution, learned time/variable relations and linear prediction head, with
explicit node roles and three directed relation masks. It does not use Sandwich
or hypernodes. See [implementation and run guide (中文)](docs/EXEN_BASELINE_zh.md)
for data roles, target-leakage tests, inference and configuration details.

```bash
pip install -r requirements-exen.txt
python -m pytest -q tests/test_exen.py
python examples/make_exen_demo.py
python main.py --config_path config/ExEn/ExEn_npz.yaml \
  --model_save_dir_path model_states/exen_demo \
  --result_save_dir_path results/exen_demo
```

To use the supplied `processed` folder directly, place it at `data/processed/`
and use `config/ExEn/processed/BE.yaml` (or another dataset's YAML).
**All raw zeros are treated as missing** in this adapter, combined with the
archive mask. See [processed-data and conda guide (中文)](docs/EXEN_PROCESSED_zh.md).

Processed-data runs now select checkpoints by DAG `mse_norm` in an independent
train-only StandardScaler space, and report raw-unit metrics separately. See
[metric spaces and DAG parity (中文)](docs/DAG_EVALUATION_zh.md).

DIST now defaults to streamed CSR legal-edge relations. See
[sparse backend, execution and benchmarks (中文)](docs/SPARSE_RELATIONS_zh.md).

Processed datasets default to all known future exogenous observations
(`known_future_exogenous_var_ids: all`). Set `[]` for a history-only ablation.
Run all datasets except DIST, writing to separate known-future output directories:

```bash
bash scripts/run_exen_future_except_dist.sh
# With an available CUDA-enabled PyTorch environment:
DEVICE=cuda:0 bash scripts/run_exen_future_except_dist.sh
```

For a matched Y-history-only ablation (removes both historical and future X),
add `--no-exogenous` to `main.py`, or run:

```bash
DEVICE=cuda:0 bash scripts/run_exen_no_exogenous_except_dist.sh
```

See [input ablations and their interpretation (中文)](docs/EXOGENOUS_ABLATION_zh.md).

## STRGCNDirect: joint relations without Sandwich

For a role-free control, `STRGCNDirect` applies the original middle-stage joint
time/variable relation and low-rank Chebyshev convolution directly to observation
nodes. It includes known future inputs, excludes query nodes as message sources,
and removes Sandwich and hypernode aggregation. ExEn experiments stay separate.
See [formulas, data and run guide (中文)](docs/STRGCN_DIRECT_zh.md).

```bash
# All 12 non-DIST datasets, with CUDA-enabled PyTorch:
DEVICE=cuda:0 bash scripts/run_strgcn_direct.sh
# Include DIST after checking its history/forecast lengths:
INCLUDE_DIST=1 DEVICE=cuda:0 bash scripts/run_strgcn_direct.sh
```

For the matched input ablation with the **same STRGCNDirect backbone**, remove
both historical and future exogenous inputs (12 non-DIST datasets):

```bash
DEVICE=cuda:0 bash scripts/run_strgcn_direct_no_exogenous.sh
```

Results go to `results/strgcn_direct_no_exogenous/`; compare with
`results/strgcn_direct_known_future/` in the same DAG standardized metric space.

## ExEnAligned: ExEn roles/stages with the same STRGCN operator

`ExEnAligned` retains the original role/status embeddings, three relation masks
and 2 temporal + 1 X-to-Y stages, while aligning the variable logit scale,
Chebyshev T0/T1 convolution, receiver-side factors and per-layer time features
with STRGCNDirect. See [operator alignment and both input modes (中文)](docs/EXEN_ALIGNED_zh.md).

```bash
# Both commands default to all 12 non-DIST datasets, using separate directories:
DEVICE=cuda:0 bash scripts/run_exen_aligned.sh
DEVICE=cuda:0 bash scripts/run_exen_aligned_no_exogenous.sh
```

## ExEnGraphTransformer: content attention with graph biases

The new model keeps ExEn's inputs, roles, causal masks, staged propagation and DAG
evaluation, replacing graph convolution with multi-head attention and FFN.
Time/variable biases are encoded in augmented Q/K; grouped SDPA avoids CSR
graph operations. See [Graphormer background, formulas and run guide (中文)](docs/EXEN_GRAPH_TRANSFORMER_zh.md).

```bash
DEVICE=cuda:0 bash scripts/run_exen_graph_transformer.sh
DEVICE=cuda:0 bash scripts/run_exen_graph_transformer_no_exogenous.sh
```

Current dataset configs use 4 temporal layers + 1 within-role variable layer + 3 X→Y layers, 4 heads,
hidden dimension 32 and dropout 0.3 (attention dropout remains 0).
Both runners exclude DIST by default and use separate output directories.
Set `ATTENTION_DTYPE=auto` for BF16/FP16 CUDA attention; the default is FP32.


## 对比实验基线

独立对比方法统一放在 [`baseline/`](baseline/README.md) 目录。已包含 [DAG](baseline/dag_irregular/README.md)、[TimeXer](baseline/timexer_irregular/README.md) 和 [CrossLinear](baseline/crosslinear_irregular/README.md)，提供 FullExo 和 Y-only NoExo 两种模式。TimeXer 与 CrossLinear 的全数据集有外生实验可运行 `bash baseline/run_full_exo.sh`。

## t-PatchGNN record-based supplementary experiments

Put the four processed datasets under `data/tpatch-data/{mimic,physionet,activity,ushcn}/processed/`.
The new adapter splits source records before constructing record-local tasks, pads
ragged observations per batch, and uses the supplied oracle future-X role indices.
Clinical tasks are 24h→24h, Activity 3s→1s, and USHCN 24 months→1 month.
See [task definitions, roles and run guide (中文)](docs/TPATCH_ORACLE_zh.md).

```bash
DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh
NO_EXOGENOUS=1 DEVICE=cuda:0 bash scripts/run_tpatch_oracle.sh
# Also supports MODEL=ExEn, ExEnAligned, or STRGCNDirect.
```

The tpatch configs now use the official batch sizes (32; USHCN 192) and
`TPatchForecastTrainer`: per-variable masked MSE/MAE over the full evaluation
set, followed by an equal-variable mean. Scores use upstream MinMax/preprocessed
spaces rather than DAG StandardScaler scores. ExEnGraphTransformer batches legal
attention groups on the CPU-built relation plan; its 4+3 layers stay unchanged.
The runner writes separate `tpatch_v2` directories. See the guide above before
comparing new and old metrics.

## Original patch backbones with DAG-style future-X fusion

The new `TPatchGNNFusion` and `HiPatchFusion` baselines retain their published
historical backbones and add the same two-layer, 64-hidden MLP Fusion used by DAG.
They use the current tpatch record splits, role indices and variable-macro metrics.
See [setup, modes, source versions and comparison scope](baseline/patch_fusion/README.md).

```bash
pip install -r baseline/patch_fusion/requirements.txt
DEVICE=cuda:0 bash scripts/run_patch_fusion.sh
BACKBONE=tpatchgnn DEVICE=cuda:0 bash scripts/run_patch_fusion.sh activity
BACKBONE=hipatch MODE=NoExo DEVICE=cuda:0 bash scripts/run_patch_fusion.sh activity
```

## DIST with variable-macro evaluation

ExEnGraphTransformer, TPatchGNNFusion and HiPatchFusion now support DIST with train-only MinMax and global variable-macro MSE/MAE/RMSE. Run all three with `DEVICE=cuda:0 bash scripts/run_dist.sh`. See [DIST instructions](docs/DIST_FORECAST_zh.md) for individual models and FullExo/HistExo/NoExo modes. Other processed datasets retain their existing evaluation configs.

Within-role X→X and Y→Y attention can be ablated independently. See [module controls](docs/WITHIN_VARIABLE_ATTENTION_zh.md).
