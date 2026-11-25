# GraFITi, CRU and LatentODE forecasting cores

These are forecasting adapters around the authors' model operators, not new
approximations of the three architectures. `cores.py` exposes query features
before the original prediction head. The shared external-variable plugin lives
in `models/exogenous/irregular_adapters.py` and is outside these cores.

## Source versions

| Model | Author repository | Pinned commit |
| --- | --- | --- |
| GraFITi | [yalavarthivk/GraFITi](https://github.com/yalavarthivk/GraFITi) | `698060bb0e9385d79eaf41b7981a05390725a1e5` |
| CRU | [boschresearch/Continuous-Recurrent-Units](https://github.com/boschresearch/Continuous-Recurrent-Units) | `79723e7772102cae8a78109e18d81beff9c7cac2` |
| LatentODE | [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode) | `c0682d4f52b806fb88d965755892eadd9783f936` |

Each vendor directory includes `UPSTREAM.json` listing copied source files.
CRU's AGPL-3.0-or-later license, inherited MIT notices and third-party notices
are preserved. LatentODE's MIT license is preserved. The pinned GraFITi tree
does not contain a root LICENSE file; no replacement license is asserted here.

## Local compatibility changes

- Imports are package-relative, avoiding collisions between authors' `lib`
  packages; trailing whitespace is normalized. No upstream dataset loader or
  training CLI is used.
- GraFITi optionally returns pre-output edge features. Its default forward and
  scalar output layer are retained; channel one-hot encodings follow the weight
  dtype. Feature exposure is checked against the native prediction exactly.
- Full CRU retains the original continuous transition and Van Loan covariance
  calculation. `geotorch` is imported only if the unused fast/orthogonal variant
  is selected. Vector hidden-layer builders reproduce upstream
  `Physionet_USHCN` from `lib/models.py`; filtering and the complete Gaussian
  decoder use float64, as upstream does.
- LatentODE retains `Encoder_z0_ODE_RNN`, the ODE-RNN/GRU update, `ODEFunc`,
  `DiffeqSolver` and linear `Decoder`. Tensor factories follow input dtype and
  device; zero-span/repeated-time recognition intervals use a zero Euler step
  instead of constructing a non-increasing solver grid. Recognition uses
  Euler; generation uses torchdiffeq's `dopri5`.

## Forecasting protocol

The repository's existing splits, scaling, variable partition and forecast
queries are shared across backbones. `MTANQueryDataset` is reused only as a
ragged-history packing helper; the three cores do not use the mTAN encoder.
Each sample is filtered/encoded independently to avoid padded time steps
changing its graph or continuous dynamics. Duplicate observations remain
distinct rows, and no future target value enters any core or plugin.

GraFITi exposes a `3 * hid_dim` feature for each query-variable pair. CRU exposes
the posterior latent mean at query times and retains its covariance decoder.
LatentODE exposes a generated latent state at each query time, integrated from
the first observed history time. CRU/LatentODE states are broadcast over target
variables for the shared plugin, followed by the corresponding original output
row. CRU fuses the mean path only; covariance is decoded from the original
filter state. The plugin uses its own dtype and is cast back before decoding.

GraFITi trains with variable-macro forecast MSE. CRU trains with variable-macro
Gaussian NLL, including its positive variance head. LatentODE uses one posterior
sample during training, forecast MSE plus `0.001 * KL(q(z0) || N(0,I))`, and the
posterior mean at evaluation. This last loss is a forecasting adaptation,
**not** the upstream importance-weighted likelihood objective. Early stopping
and reporting use forecast MSE/MAE under the existing repository protocol.
These experiments do not reproduce the original papers' tables.

See [Chinese integration and launch guide](../../docs/IRREGULAR_BACKBONE_FUSION_zh.md).
