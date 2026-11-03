
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
