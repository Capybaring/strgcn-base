"""Share the tested DAG observation selection and interpolation contract."""
import math
import torch
from torch import nn
from baseline.dag_irregular.adapter import IrregularDAG


class IrregularGridModel(nn.Module):
    # Reuse input gates, fallbacks, differentiable output interpolation verbatim.
    grids = IrregularDAG.grids
    forward = IrregularDAG.forward

    def __init__(self, target_ids, exogenous_ids, history_seconds, forecast_seconds,
                 seq_len=96, use_exogenous=True):
        super().__init__()
        if (not math.isfinite(history_seconds) or not math.isfinite(forecast_seconds)
                or min(history_seconds, forecast_seconds) <= 0
                or not isinstance(seq_len, int) or seq_len < 2):
            raise ValueError('Positive finite durations and integer seq_len >= 2 required')
        self.target_ids, self.exogenous_ids = list(target_ids), list(exogenous_ids)
        if (not self.target_ids or len(set(self.target_ids)) != len(self.target_ids)
                or len(set(self.exogenous_ids)) != len(self.exogenous_ids)
                or set(self.target_ids) & set(self.exogenous_ids)):
            raise ValueError('Target and exogenous IDs must be unique and disjoint')
        self.use_exogenous = use_exogenous
        self.boundary = history_seconds/(history_seconds+forecast_seconds)
        self.seq_len = seq_len
        step = self.boundary/seq_len
        self.pred_len = math.ceil(forecast_seconds/(history_seconds/seq_len))+1
        self.register_buffer('history_grid',torch.arange(seq_len,dtype=torch.float64)*step)
        self.register_buffer('future_grid',self.boundary+torch.arange(self.pred_len,dtype=torch.float64)*step)
        self.exo_width = max(1,len(self.exogenous_ids)) if use_exogenous else 1
        self.additional_loss = None
