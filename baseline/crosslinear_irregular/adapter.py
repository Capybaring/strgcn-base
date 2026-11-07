"""Official CrossLinear plus DAG-style pointwise future-X MLP fusion."""
from types import SimpleNamespace
import torch
from torch import nn
from baseline.common.grid import IrregularGridModel
from .vendor.model import Model


class ForecastCore(nn.Module):
    def __init__(self, config, target_count, exogenous_width, use_exogenous, fusion_hidden):
        super().__init__()
        self.model = Model(config)
        self.target_count = target_count
        self.future_fusion = (nn.Sequential(nn.Linear(target_count+exogenous_width,fusion_hidden),
            nn.ReLU(),nn.Linear(fusion_hidden,target_count)) if use_exogenous else None)

    def forward(self, history, future):
        # Official MS uses LAST channel as target. Keep all Y history and X history.
        n = self.target_count
        ordered = torch.cat([history[:,:,n:],history[:,:,:n]],dim=-1)
        output = self.model(ordered,None,None,None)[:,:,-n:]
        if self.future_fusion is not None:
            output = self.future_fusion(torch.cat([output,future],dim=-1))
        return output,0


class IrregularCrossLinear(IrregularGridModel):
    def __init__(self, *args, **params):
        grid = {k: params.pop(k) for k in ('seq_len','use_exogenous') if k in params}
        super().__init__(*args, **grid)
        defaults = dict(patch_len=16,d_model=512,d_ff=2048,alpha=1.,beta=1.,fusion_hidden=64)
        if set(params)-set(defaults):
            raise ValueError(f'Unknown CrossLinear settings: {set(params)-set(defaults)}')
        defaults.update(params)
        if min(defaults['patch_len'],defaults['d_model'],defaults['d_ff'],defaults['fusion_hidden'])<1:
            raise ValueError('Model dimensions must be positive')
        fusion_hidden = defaults.pop('fusion_hidden')
        self.model_config = dict(defaults,seq_len=self.seq_len,pred_len=self.pred_len,
            dec_in=len(self.target_ids)+self.exo_width,features='MS' if len(self.target_ids)==1 else 'M',
            task_name='long_term_forecast')
        self.core = ForecastCore(SimpleNamespace(**self.model_config),len(self.target_ids),
                                 self.exo_width,self.use_exogenous,fusion_hidden)
        self.model_config['fusion_hidden'] = fusion_hidden
