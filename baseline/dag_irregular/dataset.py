"""Existing dataset protocol, preserving float64 time and exact query timestamps."""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets.ExEnProcessedDataset import ExEnProcessedDataset


class DAGProcessedDataset(ExEnProcessedDataset):
    def __getitem__(self, index):
        p = self.packet
        a, h, b, start = p['windows'][index]
        row, var = np.where(p['mask'][a:b])
        absolute = p['time'][a:b][row]
        relative = absolute-start
        future = relative >= p['history_seconds']
        keep = ~future | np.isin(var, p['target_ids']+list(self.known_future_exogenous_var_ids))
        if not self.use_exogenous:
            keep &= np.isin(var, p['target_ids'])
        row, var, relative, absolute, future = (v[keep] for v in (row,var,relative,absolute,future))
        values = p['values'][a:b][row,var].copy()
        if 'scale_min' in p:
            values = (values-p['scale_min'][var])/p['scale_range'][var]
        return torch.tensor(np.column_stack([relative/p['duration_seconds'],values,var,future,absolute]),
                            dtype=torch.float64)

    def long_collate_fn(self, samples):
        out = super().long_collate_fn(samples)
        out['batch_timestamp'] = pad_sequence([s[:,0] for s in samples], batch_first=True)
        out['batch_absolute_timestamp'] = pad_sequence([s[:,4] for s in samples], batch_first=True)
        return out
