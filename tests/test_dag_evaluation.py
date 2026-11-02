import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from utils.dag_evaluation import DAGEvaluationScaler, window_metrics
from utils.scaler import ExEnProcessedMinMaxScaler
from datasets import ExEnProcessedDataset
from models import ExEn
from trainers import ExEnDAGForecastTrainer
from test_exen_processed import archive


def test_evaluation_scaler_fits_raw_train_observations_only(tmp_path):
    _, _, data = archive(tmp_path)
    packet = data['train_data']
    scaler = DAGEvaluationScaler().fit(packet)
    for j in range(3):
        train = np.concatenate([packet['values'][a:b,j][packet['mask'][a:b,j]]
                                for a,_,b,_ in packet['windows']])
        expected = StandardScaler().fit(train.astype('float64')[:,None])
        np.testing.assert_allclose(scaler.mean_[j], expected.mean_[0])
        np.testing.assert_allclose(scaler.scale_[j], expected.scale_[0])
    assert scaler.mean_[0] < 40  # held-out stream reaches 101
    scaler.save(tmp_path/'evaluation')
    loaded = DAGEvaluationScaler().load(tmp_path/'evaluation.npz')
    np.testing.assert_array_equal(loaded.mean_, scaler.mean_)
    np.testing.assert_array_equal(loaded.count_, scaler.count_)


def test_dag_standard_space_and_window_average_are_distinct_from_minmax_and_micro():
    scale = DAGEvaluationScaler()
    scale.mean_ = np.array([100.,1000.]);scale.scale_ = np.array([10.,100.])
    scale.names = ['Y1','Y2']
    truth = np.array([[100.,1000.,np.nan],[100.,np.nan,np.nan]])
    pred = np.array([[110.,1300.,np.nan],[150.,np.nan,np.nan]])
    ids = np.array([[0,1,0],[0,1,0]])
    mask = np.array([[1,1,0],[1,0,0]],bool)
    result,counts = window_metrics(truth,pred,mask,ids,scale)
    np.testing.assert_allclose(result['mse_norm'], [5,25])
    np.testing.assert_allclose(result['mae_norm'], [2,5])
    assert result['mse_norm'].mean() == 15
    assert result['rmse_norm'].mean() == pytest.approx((5**.5+5)/2)
    assert result['rmse_norm'].mean() != pytest.approx(result['mse_norm'].mean()**.5)
    assert result['mse'].mean() == 23775
    np.testing.assert_array_equal(counts, [2,1])  # centered targets remain valid!


def test_constant_training_channel_uses_standard_scaler_unit_scale():
    p=dict(names=['Y'],values=np.array([[7.],[7.],[999.]]),mask=np.ones((3,1),bool),windows=[(0,1,2,0.)])
    scaler=DAGEvaluationScaler().fit(p)
    assert scaler.mean_[0]==7 and scaler.scale_[0]==1 and scaler.count_[0]==2


def trainer(tmp_path, model, scale, eval_scale, global_args):
    return ExEnDAGForecastTrainer(model,torch.optim.Adam(model.parameters()),None,scale,
         str(tmp_path),str(tmp_path),max_epoch_num=1,do_inverse_scaling=False,
         global_args=global_args,evaluation_scaler=eval_scale)


def test_evaluation_independent_of_batch_size_and_exports_raw_units(tmp_path):
    _,g,data=archive(tmp_path)
    scale=ExEnProcessedMinMaxScaler();scale.fit(data['train_data'])
    eval_scale=DAGEvaluationScaler().fit(data['train_data'])
    ds=ExEnProcessedDataset(scale.transform(data['test_data']),type='test',global_args=g)
    torch.manual_seed(3);model=ExEn(global_args=g,hid_dim=8)
    t=trainer(tmp_path,model,scale,eval_scale,g)
    a=t.evaluate(DataLoader(ds,batch_size=1,collate_fn=ds.collate_fn),['mse_norm','mae_norm','rmse_norm'])
    b=t.evaluate(DataLoader(ds,batch_size=2,collate_fn=ds.collate_fn),['mse_norm','mae_norm','rmse_norm'])
    np.testing.assert_allclose(a[0],b[0],rtol=1e-6)
    np.testing.assert_allclose([v for _,v in a[1]],[v for _,v in b[1]],rtol=1e-6)
    assert t.last_eval_summary['primary_space']=='train_standardized'
    assert t.last_eval_summary['forecast_window_count']==2
    assert np.min(b[3][t.last_eval_mask])>=88  # raw held-out Y units, not [0,1]


def test_bad_metric_or_nonfinite_observed_prediction_fails():
    s=DAGEvaluationScaler();s.mean_=np.array([0.]);s.scale_=np.array([1.])
    with pytest.raises(ValueError,match='Invalid observed'):
        window_metrics(np.array([[1.]]),np.array([[np.nan]]),np.array([[1]]),np.array([[0]]),s)
