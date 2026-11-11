"""MinMax/preprocessed-space, variable-macro metrics matching t-PatchGNN."""
import warnings
import numpy as np
import torch
from tqdm import tqdm
from .IrrFCGraphForecastTrainer import IrrFCGraphForecastTrainer
from utils.tpatch_evaluation import variable_error_sums, variable_macro_loss, summarize_variable_errors


class TPatchForecastTrainer(IrrFCGraphForecastTrainer):
    def __init__(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='do_inverse_scaling is False.*')
            super().__init__(*args, **kwargs)
        if self.do_inverse_scaling or self.enable_autocast or self.enable_distributed:
            raise ValueError('TPatch trainer uses supplied scaled values; full-model AMP and DDP are not supported')
        self.evaluation_history = []

    def _batch(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _predict(self, b):
        args = {k: b[k] for k in ('batch_timestamp', 'batch_var_idx', 'batch_pred_mask',
                                  'batch_pad_mask', 'batch_time_id', 'batch_node_type') if k in b}
        args.update({k: v for k, v in b.items() if k.startswith('batch_backbone_')})
        if 'batch_attention_plan' in b:
            args['batch_attention_plan'] = b['batch_attention_plan']
        return self.model(batch_value=torch.where(b['batch_pred_mask'].bool(), 0, b['batch_value']),
                          **args).squeeze(-1)

    def train_one_epoch(self, data_loader):
        self.model.train()
        total = 0.
        iterator = tqdm(data_loader)
        for raw in iterator:
            b = self._batch(raw)
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self._predict(b)
            loss = variable_macro_loss(prediction, b['batch_value'],
                b['batch_pred_mask'].bool() & b['batch_pad_mask'].bool(),
                b['batch_var_idx'], self.global_args['num_of_vals'])
            loss.backward()
            self.optimizer.step()
            value = loss.item()
            total += value
            iterator.set_description(f'loss is {value:.4f}')
        return total / len(data_loader)

    @torch.no_grad()
    def evaluate(self, data_loader, metrics, test_flag=False, **kwargs):
        if not set(metrics) <= {'mse', 'mae', 'rmse'}:
            raise ValueError('TPatch metric names are mse, mae and rmse')
        self.model.eval()
        dim = self.global_args['num_of_vals']
        se, ae, count = (torch.zeros(dim, dtype=torch.float64, device=self.device) for _ in range(3))
        exported = {k: [] for k in ('prediction', 'actual', 'mask', 'variable_id', 'window_id')}
        windows = 0
        for raw in data_loader:
            b = self._batch(raw)
            pred = self._predict(b)
            mask = b['batch_pred_mask'].bool() & b['batch_pad_mask'].bool()
            sums = variable_error_sums(pred.double(), b['batch_value'].double(), mask,
                                       b['batch_var_idx'], dim)
            se += sums[0]; ae += sums[1]; count += sums[2]
            if test_flag:
                ids = b['batch_var_idx'].cpu().numpy()
                n, width = ids.shape
                wid = np.broadcast_to(np.arange(windows, windows + n)[:, None], (n, width))
                for key, value in zip(exported, (pred.cpu().numpy(), b['batch_value'].cpu().numpy(),
                                                 mask.cpu().numpy(), ids, wid)):
                    exported[key].append(value.reshape(-1))
            windows += b['batch_value'].shape[0]
        if windows == 0 or not count.any():
            raise ValueError('Empty evaluation or no observed targets')
        summary = summarize_variable_errors(se, ae, count)
        target_ids = data_loader.dataset.packet['target_ids']
        summary.update(loss=summary['mse'], forecast_window_count=windows,
            target_ids=target_ids, target_names=[data_loader.dataset.packet['names'][i] for i in target_ids],
            observed_count_by_target=count[target_ids].long().cpu().tolist(),
            primary_metric='mse', primary_space=('tpatch_input_space'
                if data_loader.dataset.packet.get('kind') == 'tpatch_records' else 'train_minmax_input_space'),
            aggregation='global_mean_per_variable_then_equal_mean_available_variables',
            prediction_space='model_input_space', reference='usail-hkust/t-PatchGNN/lib/evaluation.py')
        self.last_eval_summary = summary
        self.evaluation_history.append(dict(summary))
        arrays = {k: np.concatenate(v) if v else np.array([]) for k, v in exported.items()}
        if test_flag:
            self.last_eval_mask, self.last_eval_var_idx = arrays['mask'], arrays['variable_id']
            self.last_eval_window_id = arrays['window_id']
        print(f'{type(self.model).__name__} evaluation:', {k: summary[k] for k in metrics})
        return summary['mse'], [(k, summary[k]) for k in metrics], arrays['prediction'], arrays['actual']

    def test(self, *args, **kwargs):
        result, prediction, actual = super().test(*args, **kwargs)
        result.update(self.last_eval_summary)
        return result, prediction, actual
