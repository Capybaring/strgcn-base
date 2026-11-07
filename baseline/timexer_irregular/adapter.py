"""TimeXer with the DAG benchmark's future-X encoder extension."""
from types import SimpleNamespace
from torch import nn
from baseline.common.grid import IrregularGridModel
from .vendor.model import timexer_model


class ForecastCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = timexer_model(config)

    def forward(self, history, future):
        return self.model(history, future), 0


class IrregularTimeXer(IrregularGridModel):
    def __init__(self, *args, **params):
        grid = {k: params.pop(k) for k in ('seq_len','use_exogenous') if k in params}
        super().__init__(*args, **grid)
        defaults = dict(patch_len=16,d_model=512,d_ff=2048,n_heads=8,e_layers=2,
                        dropout=.2,factor=1,activation='gelu')
        if set(params)-set(defaults):
            raise ValueError(f'Unknown TimeXer settings: {set(params)-set(defaults)}')
        defaults.update(params)
        if (not 1 <= defaults['patch_len'] <= self.seq_len
                or defaults['d_model'] % defaults['n_heads'] or defaults['d_model'] % 2):
            raise ValueError('Patch must fit history; d_model must be even and divisible by n_heads')
        self.model_config = dict(defaults,seq_len=self.seq_len,pred_len=self.pred_len,
            enc_in=len(self.target_ids)+self.exo_width,series_dim=len(self.target_ids),
            use_future=self.use_exogenous,freq='h')
        self.core = ForecastCore(SimpleNamespace(**self.model_config))
