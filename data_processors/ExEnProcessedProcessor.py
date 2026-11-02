"""Read time/values/mask/names archives into chronological physical-time windows."""
from pathlib import Path
import numpy as np
from utils.parse_duration_to_seconds import parse_duration_to_seconds


class ExEnProcessedProcessor:
    def __init__(self, global_args, history_len, forecast_len, target_names,
                 split_ratios=(.6, .2, .2), max_windows_per_split=None, **kwargs):
        self.global_args = global_args
        self.path = Path(global_args['exen_data_path'])
        self.history = parse_duration_to_seconds(history_len)
        self.forecast = parse_duration_to_seconds(forecast_len)
        self.targets = list(target_names)
        self.ratios = np.asarray(split_ratios, dtype=float)
        self.limit = max_windows_per_split
        if (not np.isfinite([self.history, self.forecast]).all()
                or min(self.history, self.forecast) <= 0):
            raise ValueError('History and forecast durations must be positive and finite')
        if self.ratios.shape != (3,) or (self.ratios <= 0).any() or not np.isclose(self.ratios.sum(), 1):
            raise ValueError('split_ratios must be three positive fractions summing to one')
        if self.limit is not None and (not isinstance(self.limit, int) or self.limit < 1):
            raise ValueError('max_windows_per_split must be a positive integer')

    def process(self):
        if not self.path.exists():
            raise FileNotFoundError(f'{self.path}: put the supplied processed folder under data/processed/')
        with np.load(self.path, allow_pickle=False) as z:
            t = np.asarray(z['time'], dtype=np.float64)
            x = np.asarray(z['values'], dtype=np.float32)
            original_mask = np.asarray(z['mask'])
            names = z['names'].tolist()
        if (t.ndim != 1 or len(t) < 2 or x.ndim != 2 or x.shape != original_mask.shape
                or len(t) != len(x) or len(names) != x.shape[1] or len(set(names)) != len(names)):
            raise ValueError('Invalid time/values/mask/names shapes or duplicate variable names')
        if not np.isfinite(t).all() or not (np.diff(t) > 0).all():
            raise ValueError('time must be finite and strictly increasing')
        if not np.isin(original_mask, [0, 1]).all():
            raise ValueError('mask must contain only 0/1')
        # User's dataset rule: ALL raw zeros are missing, including originally observed zeros.
        mask = original_mask.astype(bool) & np.isfinite(x) & (x != 0)
        if not self.targets or len(set(self.targets)) != len(self.targets) or not set(self.targets) < set(names):
            raise ValueError('target_names must select some, but not all, archive variables')
        target_ids = [names.index(n) for n in self.targets]
        exo_ids = [i for i in range(len(names)) if i not in target_ids]
        duration = self.history + self.forecast
        origin = float(t[0])
        # Conservative coverage: never infer coverage beyond the last observed timestamp.
        count = int(np.floor((t[-1] - origin) / duration))
        cuts = [0, int(count * self.ratios[0]), int(count * self.ratios[:2].sum()), count]
        result, summary = {}, {}
        for s, split in enumerate(('train', 'valid', 'test')):
            windows = []
            for index in range(cuts[s], cuts[s + 1]):
                start = origin + index * duration
                a, h, b = np.searchsorted(t, [start, start + self.history, start + duration])
                if h == a or b == h or not mask[a:h].any() or not mask[h:b, target_ids].any():
                    continue
                windows.append((int(a), int(h), int(b), float(start)))
                if self.limit is not None and len(windows) >= self.limit:
                    break
            if not windows:
                raise ValueError(f'No usable {split} windows; shorten durations or provide more data')
            result[f'{split}_data'] = dict(time=t, values=x, mask=mask, names=names,
                windows=windows, history_seconds=self.history, duration_seconds=duration,
                target_ids=target_ids, exogenous_ids=exo_ids)
            summary[split] = dict(windows=len(windows), candidate_windows=cuts[s+1]-cuts[s],
                first_start=windows[0][3], last_end=windows[-1][3]+duration)
        self.global_args.update(dataset_name=self.path.stem, exen_variable_names=names,
            exen_processed_summary=summary, exen_zero_as_missing=True)
        result['seen_data'] = result['train_data']
        return result
