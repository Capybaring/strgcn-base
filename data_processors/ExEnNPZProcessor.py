"""Load pre-split long-form arrays without changing their split or variable IDs."""
import numpy as np


class ExEnNPZProcessor:
    def __init__(self, global_args, **kwargs):
        self.path = global_args['exen_data_path']

    def process(self):
        with np.load(self.path, allow_pickle=False) as archive:
            result = {}
            for split in ('train', 'valid', 'test'):
                key = f'{split}_data'
                a = np.asarray(archive[key], dtype=np.float64)
                if a.ndim != 2 or a.shape[1] != 4 or not len(a) or not np.isfinite(a).all():
                    raise ValueError(f'{key} must be nonempty finite [N,4]: series,var,time,value')
                if not np.equal(a[:, :2], np.floor(a[:, :2])).all():
                    raise ValueError('Series and variable IDs must be integers')
                result[key] = a
        result['seen_data'] = result['train_data']
        return result
