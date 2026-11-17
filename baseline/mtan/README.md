# mTAN / mTAND forecasting task adaptation

Source: https://github.com/reml-lab/mTAN

Fixed upstream commit: `7a3d536ee742f1cacb4a6d3478ac78a228d995ff`.
`vendor/models.py` contains the first 178 lines of upstream `src/models.py`,
including unmodified `multiTimeAttention`, `enc_mtan_rnn`, and `dec_mtan_rnn`.
The upstream MIT license is retained at `vendor/LICENSE`.
Original full source SHA256: `cd1b3979e3bccec2333260ca11d9fa4b2ba06925cc0ac96c566c7439401882e0`.
Vendored excerpt SHA256: `ec1ef6f400e17fbaf4fc36f4e1a368ad80e6cd57d804973b00bd6a21fbbdbca3`.

This uses the official variational encoder/decoder, not a newly invented
deterministic network called mTAND. mTAND denotes the discretized multi-time
attention representation in the paper. The paper's official tasks are primarily
interpolation and classification; our future-Y prediction is an explicit task
adaptation and is not a reproduction of its published benchmark scores.

## History and query interfaces

Historical values/masks and their real normalized timestamps go into the
upstream time-attention encoder and bidirectional GRU. Equal-time variables
share an input row, but repeated same-variable observations are retained in
separate rows. Padding is masked; observed zeros remain valid. Future Y and
future X never enter the backbone. Default Y-only history zeros all X slots;
`backbone_history: yx` is an explicit alternative.

The decoder's reference-time latent sequence passes through its original
bidirectional GRU and time attention. Before `z0_to_obs`, this yields
`Z[B,Q,2*gen_hidden]` (default 100). We broadcast Z to target queries, use the
unchanged `DynamicLagFusion` V1 algorithm, and decode each target with the
corresponding row of the original MLP head. With fusion disabled this matches
the original decoder's output on the same latent sequence, up to floating-point
rounding. No trainable target embedding is added just to expose this interface.

## Training and evaluation

- Training: one reparameterized sample from the historical latent posterior.
- Objective: existing variable-macro forecast MSE plus `0.001 * KL`, where KL is
  `0.5*(mu^2+exp(logvar)-1-logvar)` summed over latent dimensions and averaged
  over samples/reference points. This task objective is **not** the upstream
  Gaussian-NLL/IWAE interpolation objective. All three methods use the same KL.
- Evaluation: posterior mean as latent input for deterministic validation/test;
  this is not the Monte Carlo posterior predictive mean.
- Selection: validation forecast MSE, not the regularized training objective.
- Four existing record splits, target roles, scaler, prediction masks and macro
  metrics are unchanged. Default 30 epochs/5-epoch early stopping, seed 3407.
- Defaults: encoder hidden 64, latent 32, decoder hidden 50, time embedding 128,
  one time-attention head, 64 reference points on `[0,1]`. These are explicit
  engineering settings, not a claim of optimal official per-dataset settings.
- Parent patch packing is reused only for query metadata/MLP alignment; patches
  are discarded before transfer to mTAN. mTAN itself receives raw observations.

## Run

```bash
# Same dependencies as existing query-fusion methods; no torchdiffeq required.
python -m pip install -r baseline/patch_fusion/requirements.txt
bash scripts/run_query_fusion.sh mtand activity v1
# Four datasets x base/mlp/v1, sequential on GPU 0, fixed seed 3407:
bash scripts/run_mtand_base_mlp_v1.sh
```

`GPU`, `DEVICE`, `PYTHON`, `TPATCH_DATA_ROOT`, `RUN_DIR` and `DRY_RUN=1` work
as in existing shell entries. Logs, masks, record/window IDs, configs and metrics
are isolated under a newly created run directory. Existing runs are never
overwritten. CPU smoke tests do not establish full-batch CUDA performance.

## Validation (2026-11-17)

133 targeted regression tests passed, including 20 mTAN tests. The AST of all
three vendored neural-network classes matches the fixed official source.
Tests cover original-head outputs/gradients, training reparameterization/KL,
identical initial backbones across variants, all four datasets and all three
methods, future-Y and later-X isolation, zeros/duplicates/padding/empty history,
CLI ablations and all 12 CLI exports. Twelve real-data CPU runs with default
model dimensions, one epoch and two windows per split completed successfully;
base/MLP/V1 used identical test labels, masks, variables, records and windows.
See `reports/mtan_query_validation.json` and `reports/mtan_query_default_shape.csv`.
These are diagnostic runs, not full-data accuracy or GPU-memory measurements.
