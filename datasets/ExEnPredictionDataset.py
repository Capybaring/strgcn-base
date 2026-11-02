"""Role-aware adapter over the original irregular windowing/collation pipeline."""
import torch

from .ITSPredictionDataset import ITSPredictionDataset
from models.STRGCN.ExEn import HIST_EXO, HIST_ENDO, FUTURE_EXO, QUERY_ENDO


class ExEnPredictionDataset(ITSPredictionDataset):
    def __init__(self, data, type, exogenous_var_ids, known_future_exogenous_var_ids=(),
                 **kwargs):
        if kwargs.get('enable_agg_mask', False) or kwargs.get('output_style', 'long') != 'long':
            raise ValueError('ExEn requires long format without aggregation')
        self.exogenous_var_ids = set(exogenous_var_ids)
        self.known_future_exogenous_var_ids = set(known_future_exogenous_var_ids)
        if not self.known_future_exogenous_var_ids <= self.exogenous_var_ids:
            raise ValueError('Known future variables must be exogenous')
        if isinstance(data, dict):
            data = data[f'{type}_data']
        # Cache identity upstream omits data identity; disable to avoid stale splits.
        kwargs['use_cache'] = False
        super().__init__(data, type=type, **kwargs)
        if type == 'train':
            self.global_args['exen_var2col'] = dict(self.var2col)
        elif 'exen_var2col' not in self.global_args:
            raise ValueError('Construct training dataset first to establish variable mapping')
        mapping = self.global_args['exen_var2col']
        if not set(self._all_var_ids) <= set(mapping):
            raise ValueError('Validation/test contains a variable absent from training')
        if not self.exogenous_var_ids or not self.exogenous_var_ids < set(mapping):
            raise ValueError('Specify valid exogenous raw IDs and leave at least one endogenous variable')
        self.var2col = dict(mapping)
        self.global_args['num_of_vals'] = len(mapping)
        indices = sorted(mapping[v] for v in self.exogenous_var_ids)
        previous = self.global_args.get('exogenous_var_indices')
        if previous is not None and previous != indices:
            raise ValueError('Variable roles must match across splits')
        self.global_args['exogenous_var_indices'] = indices
        self._samples = [s for s in self._samples if any(
            bool(row[3]) and int(row[2]) not in self.exogenous_var_ids for row in s)]
        if not self._samples:
            raise ValueError('No windows containing future endogenous labels')

    def long_collate_fn(self, batch):
        out = super().long_collate_fn(batch)
        future = out['batch_pred_mask'].bool()
        valid = out['batch_pad_mask'].bool()
        var_idx = out['batch_var_idx']
        exo = torch.zeros_like(valid)
        known = torch.zeros_like(valid)
        for v in self.exogenous_var_ids:
            exo |= var_idx == self.var2col[v]
        for v in self.known_future_exogenous_var_ids:
            known |= var_idx == self.var2col[v]
        # Unknown future X must not be silently supplied as a known input.
        valid &= ~(future & exo & ~known)
        types = torch.full_like(var_idx, HIST_ENDO)
        types[exo & ~future] = HIST_EXO
        types[exo & future] = FUTURE_EXO
        types[~exo & future] = QUERY_ENDO
        out['batch_node_type'] = types
        out['batch_pred_mask'] = (future & ~exo & valid).float()
        out['batch_pad_mask'] = valid.float()
        out['batch_value'] = torch.where(valid, out['batch_value'], 0)
        return out
