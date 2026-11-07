"""Irregular observations -> DAG grids -> original endogenous query points."""
import math
from types import SimpleNamespace
import torch
from torch import nn
from .vendor.dag_model import DAGModel


def interpolate(t, y, query, fallback=0.):
    """Linear interpolation; endpoint hold, singleton hold, empty constant.

    Duplicate timestamps are averaged. Only caller-selected available observations
    enter here. Index selection is discrete; interpolation remains differentiable
    with respect to y (in particular the DAG forecast grid).
    """
    if t.ndim != 1 or y.shape != t.shape or query.ndim != 1:
        raise ValueError('Interpolation expects 1D time, value and query arrays')
    if not torch.isfinite(query).all():
        raise ValueError('Nonfinite query timestamps')
    keep = torch.isfinite(t) & torch.isfinite(y)
    t, y = t[keep], y[keep]
    if not len(t):
        return query.new_full(query.shape, fallback, dtype=y.dtype)
    order = t.argsort(stable=True)
    t, y = t[order], y[order]
    unique, inverse, counts = torch.unique_consecutive(t, return_inverse=True, return_counts=True)
    sums = torch.zeros_like(unique, dtype=y.dtype).scatter_add(0, inverse, y)
    t, y = unique, sums / counts
    if len(t) == 1:
        return y[0].expand_as(query)
    q = query.clamp(t[0], t[-1])
    right = torch.searchsorted(t.contiguous(), q.contiguous()).clamp(1, len(t)-1)
    left = right - 1
    weight = ((q-t[left])/(t[right]-t[left])).to(y.dtype)
    return y[left] + weight * (y[right]-y[left])


class IrregularDAG(nn.Module):
    def __init__(self, target_ids, exogenous_ids, history_seconds, forecast_seconds,
                 seq_len=96, use_exogenous=True, **params):
        super().__init__()
        if (not math.isfinite(history_seconds) or not math.isfinite(forecast_seconds)
                or min(history_seconds, forecast_seconds) <= 0 or seq_len < 2):
            raise ValueError('Positive finite durations and seq_len >= 2 required')
        self.target_ids, self.exogenous_ids = list(target_ids), list(exogenous_ids)
        if (not self.target_ids or len(set(self.target_ids)) != len(self.target_ids)
                or len(set(self.exogenous_ids)) != len(self.exogenous_ids)
                or set(self.target_ids) & set(self.exogenous_ids)):
            raise ValueError('Target and exogenous IDs must be unique and disjoint')
        self.use_exogenous = use_exogenous
        self.boundary = history_seconds / (history_seconds + forecast_seconds)
        step = self.boundary / seq_len
        pred_len = math.ceil(forecast_seconds / (history_seconds / seq_len)) + 1
        self.register_buffer('history_grid', torch.arange(seq_len, dtype=torch.float64)*step)
        self.register_buffer('future_grid', self.boundary+torch.arange(pred_len, dtype=torch.float64)*step)
        self.exo_width = max(1, len(exogenous_ids)) if use_exogenous else 1
        defaults = dict(d_model=512, d_ff=2048, n_heads=8, e_layers=2, dropout=.1,
                        factor=1, activation='gelu', patch_len=16, stride=8,
                        alpha=.2, beta=.1)
        unknown = set(params)-set(defaults)
        if unknown:
            raise ValueError(f'Unsupported DAG settings: {unknown}')
        defaults.update(params)
        if defaults['patch_len'] > seq_len or defaults['stride'] < 1:
            raise ValueError('Patch length must fit history; stride must be positive')
        config = SimpleNamespace(**defaults, seq_len=seq_len, pred_len=pred_len,
            series_dim=len(target_ids), enc_in=len(target_ids)+self.exo_width,
            use_c=True, use_t=True, use_c_exog=use_exogenous,
            use_t_exog=use_exogenous, infer_use_future=True, criterion=nn.L1Loss())
        self.core = DAGModel(config)
        self.additional_loss = None

    def grids(self, batch_value, batch_timestamp, batch_var_idx, batch_pred_mask,
              batch_pad_mask, batch_node_type):
        history, future = [], []
        for value, time, ids, pred, pad, role in zip(batch_value, batch_timestamp,
                batch_var_idx, batch_pred_mask, batch_pad_mask, batch_node_type):
            valid = pad.bool() & torch.isfinite(time)
            hg = self.history_grid.to(time)
            fg = self.future_grid.to(time)
            columns = []
            # Role AND timestamp gates prevent accidental label/role leakage.
            hist = valid & ~pred.bool() & (time >= 0) & (time < self.boundary)
            for vid in self.target_ids:
                take = hist & (ids == vid) & (role == 1)
                columns.append(interpolate(time[take], value[take], hg))
            exo, fut = [], []
            if self.use_exogenous:
                for vid in self.exogenous_ids:
                    past = hist & (ids == vid) & (role == 0)
                    available = past | (valid & ~pred.bool() & (ids == vid)
                        & (role == 2) & (time >= self.boundary) & (time < 1))
                    exo.append(interpolate(time[past], value[past], hg))
                    fut.append(interpolate(time[available], value[available], fg))
            if not exo:
                exo, fut = [torch.zeros_like(hg, dtype=value.dtype)], [torch.zeros_like(fg, dtype=value.dtype)]
            history.append(torch.stack(columns+exo, -1))
            future.append(torch.stack(fut, -1))
        return torch.stack(history), torch.stack(future)

    def forward(self, batch_value, batch_timestamp, batch_var_idx, batch_pred_mask,
                batch_pad_mask, batch_node_type, **unused):
        inputs, future = self.grids(batch_value, batch_timestamp, batch_var_idx,
            batch_pred_mask, batch_pad_mask, batch_node_type)
        output, self.additional_loss = self.core(inputs, future)
        rows = []
        for b in range(len(output)):
            row = output.new_zeros(batch_value.shape[1])
            for column, vid in enumerate(self.target_ids):
                take = batch_pad_mask[b].bool() & batch_pred_mask[b].bool() & (batch_var_idx[b] == vid)
                q = batch_timestamp[b, take]
                if ((q < self.boundary-1e-7) | (q >= 1) | ~torch.isfinite(q)).any():
                    raise ValueError('Endogenous queries must lie in the forecast interval [origin,end)')
                row[take] = interpolate(self.future_grid.to(q), output[b, :, column], q)
            rows.append(row)
        return torch.stack(rows).unsqueeze(-1)
