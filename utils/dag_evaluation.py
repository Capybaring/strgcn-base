"""DAG benchmark metric spaces, extended to masked irregular forecast windows.

Reference: decisionintelligence/DAG @0758990, forecasting.py::_get_eval_scaler,
regression_metrics.py and rolling_forecast.py. Statistics use training data only.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from .packet_observations import raw_observations

REFERENCE = 'decisionintelligence/DAG@0758990'
METRICS = ('mse_norm', 'mae_norm', 'rmse_norm', 'mse', 'mae', 'rmse')


class DAGEvaluationScaler:
    """Independent from model-input MinMax scaling; raw observed cells only."""
    def fit(self, packet):
        dim = len(packet['names'])
        self.mean_ = np.empty(dim, dtype=np.float64)
        self.scale_ = np.empty(dim, dtype=np.float64)
        self.count_ = np.zeros(dim, dtype=np.int64)
        self.names = list(packet['names'])
        self.fit_scope = 'training_records_only' if packet.get('kind') == 'tpatch_records' else 'training_windows_only'
        self.zero_rule = ('original_mask & finite; observed zero retained; invalid PhysioNet negatives masked'
                          if packet.get('kind') == 'tpatch_records' else
                          'raw zero is missing; standardized zero remains valid')
        for j in range(dim):
            pieces = [values[:, j][mask[:, j]] for values, mask in raw_observations(packet)]
            observed = np.concatenate(pieces).astype(np.float64)
            if not len(observed) and packet.get('kind') == 'tpatch_records':
                self.mean_[j], self.scale_[j] = 0., 1.
                continue
            if not len(observed) or not np.isfinite(observed).all():
                raise ValueError(f'Invalid or empty evaluation-scaler training data: {self.names[j]}')
            fitted = StandardScaler().fit(observed[:, None])
            self.mean_[j], self.scale_[j] = fitted.mean_[0], fitted.scale_[0]
            self.count_[j] = len(observed)
        return self

    def save(self, path):
        np.savez(path, mean=self.mean_, scale=self.scale_, count=self.count_, names=self.names,
                 fit_scope=self.fit_scope, zero_rule=self.zero_rule)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.mean_, self.scale_, self.count_ = data['mean'], data['scale'], data['count']
            self.names = data['names'].tolist()
            self.fit_scope = str(data['fit_scope']) if 'fit_scope' in data else 'training_windows_only'
            self.zero_rule = str(data['zero_rule']) if 'zero_rule' in data else 'raw zero is missing; standardized zero remains valid'
        return self

    def manifest(self):
        return dict(type='StandardScaler', fit_scope=getattr(self, 'fit_scope', 'training_windows_only'), ddof=0,
                    mean=self.mean_.tolist(), scale=self.scale_.tolist(),
                    observed_count=self.count_.tolist(), variable_names=self.names,
                    zero_rule=getattr(self, 'zero_rule', 'raw zero is missing; standardized zero remains valid'))


def window_metrics(actual, predicted, mask, var_idx, scaler):
    """Inputs are RAW units [B,L]; each row is one forecast window.

    DAG averages over the horizon and target channels within each window, then
    averages window metrics. Missing cells are excluded in this irregular adapter.
    """
    arrays = [np.asarray(v) for v in (actual, predicted, mask, var_idx)]
    actual, predicted, mask, var_idx = arrays
    if actual.ndim != 2 or any(a.shape != actual.shape for a in arrays):
        raise ValueError('All metric arrays must share [batch, nodes] shape')
    values = {k: [] for k in METRICS}
    observed = np.zeros(len(scaler.mean_), dtype=np.int64)
    for truth, pred, valid, ids in zip(actual, predicted, mask, var_idx):
        valid = valid.astype(bool)
        truth, pred, ids = truth[valid].astype(np.float64), pred[valid].astype(np.float64), ids[valid]
        if not len(truth):
            raise ValueError('Forecast window has no observed targets')
        if (not np.isfinite(truth).all() or not np.isfinite(pred).all()
                or not np.issubdtype(ids.dtype, np.integer)
                or (ids < 0).any() or (ids >= len(scaler.mean_)).any()):
            raise ValueError('Invalid observed target, prediction, or variable index')
        # Explicit transform of raw units, exactly as DAG's _error_norm does.
        z_true = (truth - scaler.mean_[ids]) / scaler.scale_[ids]
        z_pred = (pred - scaler.mean_[ids]) / scaler.scale_[ids]
        raw_error, norm_error = truth - pred, z_true - z_pred
        for suffix, error in (('', raw_error), ('_norm', norm_error)):
            mse = float(np.mean(error ** 2))
            values['mse' + suffix].append(mse)
            values['mae' + suffix].append(float(np.mean(np.abs(error))))
            values['rmse' + suffix].append(float(np.sqrt(mse)))
        observed += np.bincount(ids, minlength=len(observed))
    return {k: np.asarray(v, dtype=np.float64) for k, v in values.items()}, observed
