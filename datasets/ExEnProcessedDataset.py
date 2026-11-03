"""Lazy long-form windows: keep the original dense archive compact in memory."""
import numpy as np
import torch
from .ExEnPredictionDataset import ExEnPredictionDataset


class ExEnProcessedDataset(ExEnPredictionDataset):
    def __init__(self, data, type, global_args, known_future_exogenous_var_ids="all", use_exogenous=True, **kwargs):
        self.packet = data[f'{type}_data'] if f'{type}_data' in data else data
        self.global_args = global_args
        self.type = type
        if not isinstance(use_exogenous, bool):
            raise ValueError("use_exogenous must be true or false")
        self.use_exogenous = use_exogenous
        self.enable_agg_mask = False
        self.output_style = 'long'
        names = self.packet['names']
        self.var2col = {i: i for i in range(len(names))}
        self.exogenous_var_ids = set(self.packet['exogenous_ids'])
        if isinstance(known_future_exogenous_var_ids, str):
            if known_future_exogenous_var_ids != 'all':
                raise ValueError('known_future_exogenous_var_ids must be all or a list of IDs')
            self.known_future_exogenous_var_ids = set(self.exogenous_var_ids)
        else:
            self.known_future_exogenous_var_ids = set(known_future_exogenous_var_ids)
        if not self.known_future_exogenous_var_ids <= self.exogenous_var_ids:
            raise ValueError('Known future variable IDs must be exogenous archive column indices')
        if not self.use_exogenous:
            self.known_future_exogenous_var_ids = set()
        global_args['num_of_vals'] = len(names)
        global_args['exogenous_var_indices'] = sorted(self.exogenous_var_ids)
        global_args['exen_var2col'] = self.var2col

    def __len__(self):
        return len(self.packet['windows'])

    def __getitem__(self, index):
        p = self.packet
        a, h, b, start = p['windows'][index]
        row, var = np.where(p['mask'][a:b])
        times = p['time'][a:b][row] - start
        future = times >= p['history_seconds']
        # Remove unavailable future X before padding/dense graph construction.
        keep = ~future | np.isin(var, p['target_ids'] + list(self.known_future_exogenous_var_ids))
        if not self.use_exogenous:
            keep &= np.isin(var, p["target_ids"])
        row, var, times, future = row[keep], var[keep], times[keep], future[keep]
        values = p['values'][a:b][row, var].copy()
        if 'scale_min' in p:
            values = (values - p['scale_min'][var]) / p['scale_range'][var]
        sample = np.column_stack([times / p['duration_seconds'], values, var, future])
        # Zeros produced by scaling are VALID; missingness was decided on raw values.
        return torch.tensor(sample, dtype=torch.float32)
