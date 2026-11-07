"""Repository checkpoint/evaluation lifecycle with the original DAG objective."""
import numpy as np
import torch
from trainers.ExEnDAGForecastTrainer import ExEnDAGForecastTrainer


class DAGIrregularTrainer(ExEnDAGForecastTrainer):
    def train_one_epoch(self, data_loader):
        self.model.train()
        losses = []
        for raw in data_loader:
            batch = {k: v.to(self.device) for k, v in raw.items()}
            inputs = dict(batch)
            inputs['batch_value'] = torch.where(batch['batch_pred_mask'].bool(), 0, batch['batch_value'])
            prediction = self.model(**inputs).squeeze(-1)
            valid = batch['batch_pred_mask'].bool() & batch['batch_pad_mask'].bool()
            if not valid.any():
                raise ValueError('No real supervised queries')
            # Preserve DAG's MAE + beta * (TC exogenous MAE + CC reconstruction MAE).
            loss = (prediction[valid]-batch['batch_value'][valid]).abs().mean() + self.model.additional_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().item())
        if not losses:
            raise ValueError('Empty training loader')
        return float(np.mean(losses))

    def evaluate(self, data_loader, metrics, **kwargs):
        _, scores, prediction, actual = super().evaluate(data_loader, metrics, **kwargs)
        valid = self.last_eval_mask.astype(bool)
        ids = self.last_eval_var_idx[valid]
        # Upstream validates MAE without auxiliary loss (model.eval()).
        # Use input-scaling units, with upstream equal batch weighting.
        error = np.abs(prediction[valid]-actual[valid]) / self.scaler.ranges[ids]
        windows = self.last_eval_window_id[valid]
        batch_size = data_loader.batch_size
        batch_losses = [error[(windows//batch_size)==i].mean()
                        for i in np.unique(windows//batch_size)]
        loss = float(np.mean(batch_losses))
        self.last_eval_summary.update(loss=loss, selection_metric='query_mae_input_scaled')
        self.evaluation_history[-1] = dict(self.last_eval_summary)
        return loss, scores, prediction, actual
