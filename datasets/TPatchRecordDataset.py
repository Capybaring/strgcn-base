"""Ragged observation-graph samples from record-local t-PatchGNN tasks."""
import numpy as np
import torch
from .ExEnProcessedDataset import ExEnProcessedDataset


class TPatchRecordDataset(ExEnProcessedDataset):
    def __init__(self, *args, pack_attention=False, attention_score_budget=1048576, within_relations=(), **kwargs):
        self.pack_attention = pack_attention
        self.attention_score_budget = attention_score_budget
        self.within_relations = tuple(within_relations)
        super().__init__(*args, **kwargs)

    def long_collate_fn(self, batch):
        out = super().long_collate_fn(batch)
        if self.pack_attention and self.rolling_window_size is None:
            from models.STRGCN.PackedAttention import pack_attention_plan
            out['batch_attention_plan'] = pack_attention_plan(out['batch_timestamp'],
                out['batch_var_idx'], out['batch_node_type'], out['batch_pad_mask'].bool(),
                max_score_elements=self.attention_score_budget, within_relations=self.within_relations)
        return out

    def __len__(self):
        return len(self.packet['windows'])

    def __getitem__(self, index):
        p = self.packet
        record, a, h, b, start = p['windows'][index]
        _, t, x, valid = p['records'][record]
        row, var = np.where(valid[a:b])
        times = t[a:b][row]
        future = times >= start + p['history_native']
        keep = np.isin(var, p['target_ids']) | (
            self.use_exogenous & np.isin(var, list(self.exogenous_var_ids))
            & (~future | np.isin(var, list(self.known_future_exogenous_var_ids))))
        # Static fields are baseline conditions, never subsequent measurements.
        baselines = p['baseline_ids']
        if baselines:
            keep &= ~np.isin(var, baselines)
        row, var, times, future = row[keep], var[keep], times[keep], future[keep]
        values = x[a:b][row, var].copy()
        if self.use_exogenous and baselines:
            baseline_row = np.flatnonzero(t == 0)
            if len(baseline_row):
                r = baseline_row[0]
                for j in baselines:
                    if not valid[r, j]:
                        continue
                    # One historical baseline and one future copy represent the same
                    # fixed condition. All queries can attend to it without expansion
                    # at every individual irregular timestamp.
                    ts = [start]
                    if j in self.known_future_exogenous_var_ids:
                        ts.append(start + p['history_native'])
                    values = np.append(values, [x[r, j]] * len(ts))
                    var = np.append(var, [j] * len(ts))
                    times = np.append(times, ts)
                    future = np.append(future, [False] + ([True] if len(ts) == 2 else []))
        if 'scale_min' in p:
            values = (values - p['scale_min'][var]) / p['scale_range'][var]
        origin = 0 if p['preserve_record_time'] else start
        sample = np.column_stack([(times - origin) / p['time_scale'], values, var, future])
        sample = sample[np.argsort(sample[:, 0], kind='stable')]
        scale = p['time_scale']
        return self._with_rolling_bounds(sample, (start - origin) / scale,
            (start + p['history_native'] - origin) / scale,
            (start + p['duration_native'] - origin) / scale)
