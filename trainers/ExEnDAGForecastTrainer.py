"""Evaluate ExEn in DAG's train-StandardScaler space and raw physical units."""
import warnings
import numpy as np
import torch
from .IrrFCGraphForecastTrainer import IrrFCGraphForecastTrainer
from utils.dag_evaluation import METRICS, REFERENCE, window_metrics


class ExEnDAGForecastTrainer(IrrFCGraphForecastTrainer):
    def __init__(self, *args, evaluation_scaler=None, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="do_inverse_scaling is False.*")
            super().__init__(*args, **kwargs)
        if evaluation_scaler is None:
            raise ValueError('DAG evaluation requires an independent train-fitted StandardScaler')
        if self.do_inverse_scaling:
            raise ValueError('DAG trainer manages both metric spaces; set do_inverse_scaling=false')
        if self.scaler is None or not hasattr(self.scaler, 'inverse_transform_mx'):
            raise ValueError('Model-input scaler must support restoring raw target units')
        self.evaluation_scaler = evaluation_scaler
        self.evaluation_history = []

    @torch.no_grad()
    def evaluate(self, data_loader, metrics, **kwargs):
        if self.enable_distributed:
            raise NotImplementedError('DAG window evaluation is not implemented for distributed execution')
        if self.enable_autocast:
            raise ValueError('DAG metric validation currently requires enable_autocast=false')
        if any(k not in METRICS for k in metrics):
            raise ValueError(f'DAG metric names must be selected from {METRICS}')
        self.model.eval()
        sums = {k: 0. for k in METRICS}
        window_count = 0
        counts = np.zeros(len(self.evaluation_scaler.mean_), dtype=np.int64)
        exported = {k: [] for k in ('prediction', 'actual', 'mask', 'variable_id', 'window_id')}
        for raw_batch in data_loader:
            batch = {k: v.to(self.device) for k, v in raw_batch.items()}
            inputs = torch.where(batch['batch_pred_mask'].bool(), 0, batch['batch_value'])
            prediction = self.model(batch_value=inputs,
                batch_timestamp=batch['batch_timestamp'], batch_var_idx=batch['batch_var_idx'],
                batch_pred_mask=batch['batch_pred_mask'], batch_pad_mask=batch['batch_pad_mask'],
                batch_time_id=batch.get('batch_time_id'), batch_node_type=batch['batch_node_type']).squeeze(-1)
            ids = batch['batch_var_idx'].cpu().numpy()
            mask = (batch['batch_pred_mask'].bool() & batch['batch_pad_mask'].bool()).cpu().numpy()
            actual_raw = self.scaler.inverse_transform_mx(batch['batch_value'].cpu().numpy(), ids)
            predicted_raw = self.scaler.inverse_transform_mx(prediction.cpu().numpy(), ids)
            per_window, observed = window_metrics(actual_raw, predicted_raw, mask, ids, self.evaluation_scaler)
            for name in METRICS:
                sums[name] += per_window[name].sum()
            counts += observed
            n, width = ids.shape
            window_ids = np.broadcast_to(np.arange(window_count, window_count+n)[:, None], (n, width))
            for name, value in zip(exported, (predicted_raw, actual_raw, mask, ids, window_ids)):
                exported[name].append(value.reshape(-1))
            window_count += n
        if not window_count:
            raise ValueError('Empty evaluation loader')
        summary = {name: float(total / window_count) for name, total in sums.items()}
        packet = getattr(data_loader.dataset, 'packet', None)
        target_ids = packet['target_ids'] if packet is not None else np.flatnonzero(counts).tolist()
        summary.update(loss=summary['mse_norm'], forecast_window_count=window_count,
                       observed_count_by_target=counts[target_ids].tolist(), target_ids=target_ids,
                       target_names=[self.evaluation_scaler.names[i] for i in target_ids],
                       primary_metric='mse_norm', primary_space='train_standardized',
                       aggregation='mean_of_forecast_window_metrics', prediction_space='original_units',
                       reference=REFERENCE)
        self.last_eval_summary = summary
        self.evaluation_history.append(dict(summary))
        arrays = {name: np.concatenate(pieces) for name, pieces in exported.items()}
        self.last_eval_mask, self.last_eval_var_idx = arrays['mask'], arrays['variable_id']
        self.last_eval_window_id = arrays['window_id']
        print('DAG evaluation:', {k: summary[k] for k in metrics})
        return summary['mse_norm'], [(k, summary[k]) for k in metrics], arrays['prediction'], arrays['actual']

    def test(self, *args, **kwargs):
        result, prediction, actual = super().test(*args, **kwargs)
        result.update(self.last_eval_summary)
        return result, prediction, actual
