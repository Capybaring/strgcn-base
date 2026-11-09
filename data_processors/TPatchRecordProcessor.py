"""t-PatchGNN .pt records: split entities first, then construct record-local tasks.

No download, raw-file rewriting, cache, global concatenation, or interpolation.
Times retain native units (hours / milliseconds / months).
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


TASKS = {
    'mimic': (96, 24., 24., 'hours'),
    'physionet': (41, 24., 24., 'hours'),
    'activity': (12, 3000., 1000., 'milliseconds'),
    'ushcn': (5, 24., 1., 'months'),
}


class TPatchRecordProcessor:
    def __init__(self, global_args, dataset, data_root='data/tpatch-data',
                 roles_path=None, split_seed=42, max_records=None,
                 max_samples_per_split=None, **kwargs):
        if dataset not in TASKS:
            raise ValueError(f'dataset must be one of {list(TASKS)}')
        self.global_args, self.dataset = global_args, dataset
        self.root = Path(global_args.get('tpatch_data_root', data_root))
        role_name = 'physionet2012' if dataset == 'physionet' else dataset
        self.roles_path = Path(roles_path) if roles_path else (
            Path(__file__).resolve().parents[1] / 'config/tpatch/roles' / f'{role_name}.indices.json')
        self.seed, self.limit, self.sample_limit = split_seed, max_records, max_samples_per_split
        for value in (max_records, max_samples_per_split):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError('Record/sample limits must be positive integers')

    def _paths(self):
        folder = self.root / self.dataset / 'processed'
        if self.dataset == 'physionet':
            return [folder / f'set-{s}_0.0.pt' for s in 'abc']
        return [folder / {'mimic': 'mimic.pt', 'activity': 'data.pt', 'ushcn': 'ushcn.pt'}[self.dataset]]

    def _load(self, dim):
        records = []
        paths = self._paths()
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f'{path}: expected t-PatchGNN processed list of records')
            # Tensor/primitive-only loader: do not fall back to unsafe pickle execution.
            raw = torch.load(path, map_location='cpu', weights_only=True)
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f'{path}: expected list[(record_id, times, values, mask)]')
            for item in raw:
                if not isinstance(item, (tuple, list)) or len(item) != 4:
                    raise ValueError(f'{path}: invalid record structure')
                rid, t, x, mask = item
                if not all(isinstance(a, torch.Tensor) for a in (t, x, mask)):
                    raise ValueError(f'{path}: times, values and mask must be tensors')
                t = t.detach().cpu().numpy().astype(np.float64)
                x = x.detach().cpu().numpy().astype(np.float32, copy=False)
                original = mask.detach().cpu().numpy()
                if (t.ndim != 1 or not len(t) or x.shape != (len(t), dim)
                        or original.shape != x.shape or not np.isfinite(t).all()
                        or (t < 0).any() or not np.isin(original, [0, 1]).all()):
                    raise ValueError(f'{path}: invalid record shapes, times or binary mask')
                # Activity can contain timestamp reversals and nonconsecutive
                # repeated timestamps. Sort stably for searchsorted task slicing;
                # retain individual observations rather than inventing aggregates.
                if (np.diff(t) < 0).any():
                    order = np.argsort(t, kind='stable')
                    t, x, original = t[order], x[order], original[order]
                valid = original.astype(bool) & np.isfinite(x)
                # Preserve measured zero (including Gender=0, MechVent=0 and coordinates).
                # PhysioNet has known negative sentinels and impossible negative temperatures.
                if self.dataset == 'physionet':
                    valid &= x >= 0
                    valid[:, 1] &= np.isin(x[:, 1], [0, 1])
                    valid[:, 2] &= x[:, 2] > 0
                    valid[:, 3] &= np.isin(x[:, 3], [1, 2, 3, 4])
                records.append((str(rid), t, x, valid))
        if len({r[0] for r in records}) != len(records):
            raise ValueError('Record IDs must be unique across the concatenated dataset')
        if self.limit is not None:
            records = records[:self.limit]
        if len(records) < 5:
            raise ValueError('Need at least five source records for the 60/20/20 split')
        return records, paths

    def _windows(self, records, history, forecast, targets):
        windows, rejected = [], 0
        for i, (_, t, _, mask) in enumerate(records):
            if self.dataset == 'activity':
                # Exact upstream Activity_time_chunk range; includes its partial last task.
                end = int(t[-1])
                starts = range(0, end - int(history), int(forecast))
            elif self.dataset == 'ushcn':
                starts = range(0, 48 - int(history) - int(forecast) + 1, int(forecast))
            else:
                starts = [0]
            for start in starts:
                stop = start + history + forecast
                inclusive = (self.dataset == 'activity' and stop >= int(t[-1])) or (
                    self.dataset == 'ushcn' and stop == 48) or self.dataset in ('mimic', 'physionet')
                a = int(np.searchsorted(t, start, side='left'))
                h = int(np.searchsorted(t, start + history, side='left'))
                b = int(np.searchsorted(t, stop, side='right' if inclusive else 'left'))
                if a == h or h == b or not mask[a:h].any() or not mask[h:b, targets].any():
                    rejected += 1
                    continue
                windows.append((i, a, h, b, float(start)))
                if self.sample_limit is not None and len(windows) >= self.sample_limit:
                    return windows, rejected
        return windows, rejected

    def process(self):
        dim, history, forecast, unit = TASKS[self.dataset]
        roles = json.loads(self.roles_path.read_text())
        variables = roles['variables']
        if roles.get('index_base') != 0 or [v['index'] for v in variables] != list(range(dim)):
            raise ValueError('Role file must align with all original feature indices, index_base=0')
        exo, endo = roles['exogenous_indices'], roles['endogenous_indices']
        if sorted(exo + endo) != list(range(dim)) or set(exo) & set(endo):
            raise ValueError('Roles must form a complete disjoint feature partition')
        for v in variables:
            if v['role'] != ('exogenous' if v['index'] in exo else 'endogenous'):
                raise ValueError('Role names disagree with role index arrays')
        # Keep unresolved original columns in the schema, but do not silently train on them.
        excluded = [52, 76] if self.dataset == 'mimic' else []
        targets = [j for j in endo if j not in excluded]
        if not targets or not exo:
            raise ValueError('Task requires both nonempty target and conditioning groups')
        records, paths = self._load(dim)
        seen, test = train_test_split(records, train_size=.8, random_state=self.seed, shuffle=True)
        train, valid = train_test_split(seen, train_size=.75, random_state=self.seed, shuffle=False)
        result, summary = {}, {}
        for split, subset in zip(('train', 'valid', 'test'), (train, valid, test)):
            windows, rejected = self._windows(subset, history, forecast, targets)
            if not windows:
                raise ValueError(f'No usable {split} tasks with historical observations and future targets')
            result[f'{split}_data'] = dict(kind='tpatch_records', records=subset, windows=windows,
                names=[v['variable'] for v in variables], target_ids=targets, exogenous_ids=exo,
                baseline_ids=[0, 1, 2, 3] if self.dataset == 'physionet' else [],
                history_native=history, forecast_native=forecast, duration_native=history + forecast,
                time_unit=unit, time_scale=48. if self.dataset == 'ushcn' else history + forecast,
                preserve_record_time=self.dataset == 'ushcn', zero_as_missing=False)
            summary[split] = dict(source_records=len(subset), tasks=len(windows),
                                  skipped_empty_tasks=rejected)
        result['seen_data'] = result['train_data']
        self.global_args.update(dataset_name=self.dataset,
            exen_variable_names=[v['variable'] for v in variables], exen_processed_summary=summary,
            exen_zero_as_missing=False, tpatch_protocol=dict(split_seed=self.seed,
                split='80% seen/20% test shuffled, then first 75% seen train/25% valid; split before chunking',
                history=history, forecast=forecast, time_unit=unit,
                source_files=[str(p) for p in paths], roles_file=str(self.roles_path),
                condition='oracle_true_future_X_plus_valid_baseline',
                excluded_target_ids=excluded, normalization_fit='train_records_only',
                record_time_order='stable sort; duplicate-time observations retained',
                unit_of_split='source_record (not subject-grouped for Activity, matching t-PatchGNN)'))
        return result
