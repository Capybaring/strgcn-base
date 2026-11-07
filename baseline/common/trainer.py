"""Repository evaluation/checkpoints with each baseline's own query objective."""
import numpy as np
import torch
from torch import nn
from trainers.ExEnDAGForecastTrainer import ExEnDAGForecastTrainer


class GridForecastTrainer(ExEnDAGForecastTrainer):
    def __init__(self,*args,objective='mse',**kwargs):
        super().__init__(*args,**kwargs)
        self.objective = objective
        self.criterion = {'mse':nn.MSELoss(),'mae':nn.L1Loss(),
                          'huber':nn.HuberLoss(delta=.5)}[objective]

    def train_one_epoch(self,data_loader):
        self.model.train()
        losses=[]
        for raw in data_loader:
            batch={k:v.to(self.device) for k,v in raw.items()}
            inputs=dict(batch)
            inputs['batch_value']=torch.where(batch['batch_pred_mask'].bool(),0,batch['batch_value'])
            prediction=self.model(**inputs).squeeze(-1)
            valid=batch['batch_pred_mask'].bool() & batch['batch_pad_mask'].bool()
            if not valid.any():raise ValueError('No real supervised queries')
            loss=self.criterion(prediction[valid],batch['batch_value'][valid])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            self.optimizer.zero_grad();loss.backward();self.optimizer.step()
            losses.append(loss.detach().item())
        if not losses:raise ValueError('Empty training loader')
        return float(np.mean(losses))

    def evaluate(self,data_loader,metrics,**kwargs):
        _,scores,prediction,actual=super().evaluate(data_loader,metrics,**kwargs)
        valid=self.last_eval_mask.astype(bool)
        ids=self.last_eval_var_idx[valid]
        error=np.abs(prediction[valid]-actual[valid])/self.scaler.ranges[ids]
        if self.objective=='mse':error=error**2
        elif self.objective=='huber':error=np.where(error<=.5,.5*error**2,.5*(error-.25))
        groups=self.last_eval_window_id[valid]//data_loader.batch_size
        loss=float(np.mean([error[groups==i].mean() for i in np.unique(groups)]))
        self.last_eval_summary.update(loss=loss,selection_metric=f'query_{self.objective}_input_scaled')
        self.evaluation_history[-1]=dict(self.last_eval_summary)
        return loss,scores,prediction,actual
