from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
import yaml
from data_processors import ExEnProcessedProcessor
from datasets import ExEnProcessedDataset
from utils.scaler import ExEnProcessedMinMaxScaler
from models import ExEn
from test_exen import graph, model


def archive(tmp_path):
    t = np.arange(101, dtype=float)
    x = np.column_stack([t + 1, t + 2, t + 3]).astype('float32')
    x[0, 0] = 0  # valid according to original mask, missing under the user's rule
    x[3, 1] = 0
    x[5, 2] = 0
    x[7, 1] = np.nan
    mask = np.ones_like(x, dtype=bool)
    mask[2, 0] = False
    p = tmp_path / 'data.npz'
    np.savez(p, time=t, values=x, mask=mask, names=['Y', 'X1', 'X2'])
    g = {'device': 'cpu', 'exen_data_path': str(p)}
    processed = ExEnProcessedProcessor(g, history_len=7, forecast_len=3, target_names=['Y']).process()
    return p, g, processed


def test_zero_mask_and_chronological_windows(tmp_path):
    path, g, data = archive(tmp_path)
    p = data['train_data']
    assert not p['mask'][0, 0] and not p['mask'][3, 1] and not p['mask'][2, 0]
    assert not p['mask'][7, 1]
    assert [len(data[k]['windows']) for k in ('train_data','valid_data','test_data')] == [6,2,2]
    for left, right in [('train_data','valid_data'),('valid_data','test_data')]:
        assert data[left]['windows'][-1][3] + 10 <= data[right]['windows'][0][3]
    with np.load(path) as raw:
        assert raw['values'][0, 0] == 0 and raw['mask'][0, 0]  # source unmodified
    ds = ExEnProcessedDataset(data, type='train', global_args=g)
    first = ds[0]
    assert not (first[:, 1] == 0).any()
    assert not ((first[:, 0] == 0) & (first[:, 2] == 0)).any()
    b = ds.collate_fn([first])
    assert not ((b['batch_node_type'] == 2) & b['batch_pad_mask'].bool()).any()
    assert int(b['batch_pred_mask'].sum()) == 3


def test_train_scaler_ignores_raw_zero_and_preserves_scaled_zero(tmp_path):
    _, g, data = archive(tmp_path)
    scaler = ExEnProcessedMinMaxScaler()
    scaler.fit({'train_data':data['train_data']})
    assert scaler.minimum[0] == 2 and scaler.ranges[0] == 58
    p = scaler.transform({'train_data':data['train_data']})
    assert p['values'] is data['train_data']['values']  # no full-matrix copies
    ds = ExEnProcessedDataset(p, type='train', global_args=g)
    sample = ds[0]
    assert (sample[:, 1] == 0).any()  # minimum raw nonzero observation remains valid
    restored = scaler.inverse_transform_mx(np.array([0.,1.]), np.array([0,0]))
    np.testing.assert_allclose(restored, [2,60])
    scaler.save(tmp_path/'scale')
    other = ExEnProcessedMinMaxScaler();other.load(tmp_path/'scale.npz')
    np.testing.assert_allclose(other.minimum, scaler.minimum)


@pytest.mark.parametrize('known', [[], [1,2]])
def test_known_future_and_real_training(tmp_path, known):
    _, g, data = archive(tmp_path)
    scale = ExEnProcessedMinMaxScaler();scale.fit(data['train_data'])
    ds = ExEnProcessedDataset(scale.transform(data['train_data']), type='train',global_args=g,
                             known_future_exogenous_var_ids=known)
    b = ds.collate_fn([ds[0], ds[1]])
    assert bool(((b['batch_node_type']==2)&b['batch_pad_mask'].bool()).any()) == bool(known)
    m=ExEn(global_args=g,hid_dim=8,receiver_chunk_size=4)
    q=b['batch_pred_mask'].bool()
    pred=m(**b).squeeze(-1)
    loss=(pred[q]-b['batch_value'][q]).square().mean();loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


@pytest.mark.parametrize('chunk_size', [1, 3, 100])
def test_chunked_values_and_gradients_match_dense(chunk_size):
    dense=model();chunked=model(receiver_chunk_size=chunk_size)
    chunked.load_state_dict(dense.state_dict())
    b=graph()
    a=dense(**b);c=chunked(**b)
    torch.testing.assert_close(a,c,atol=1e-6,rtol=1e-5)
    a[:,4:6].sum().backward();c[:,4:6].sum().backward()
    for (name,p),(other,q) in zip(dense.named_parameters(),chunked.named_parameters()):
        assert name==other
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-4)
    dense.eval();chunked.eval()
    with torch.no_grad():torch.testing.assert_close(dense(**b),chunked(**b))


@pytest.mark.parametrize("backend", ["dense", "sparse_csr"])
def test_processed_cli(tmp_path, backend):
    path, _, _ = archive(tmp_path)
    root=Path(__file__).resolve().parents[1]
    c=yaml.safe_load((root/'config/ExEn/processed/BE.yaml').read_text())
    c['global_config']['exen_data_path']=str(path)
    c['model_config']['model_params']['relation_backend']=backend
    c['data_config']['data_processor_params'].update(history_len=7,forecast_len=3,target_names=['Y'])
    c['train_config']['trainer_params'].update(max_epoch_num=1,do_inverse_scaling=False)
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(c))
    run=subprocess.run([sys.executable,'main.py','--config_path',str(p),
       '--model_save_dir_path',str(tmp_path/'model'),'--result_save_dir_path',str(tmp_path/'result')],
       cwd=root,text=True,capture_output=True,timeout=60)
    assert run.returncode==0,run.stdout+run.stderr
    assert (tmp_path/'result/result.json').exists()
    mask=np.load(tmp_path/'result/test_y_mask.npy')
    ids=np.load(tmp_path/'result/test_var_idx.npy')
    assert mask.sum()>0 and (ids[mask.astype(bool)]==0).all()
    import json
    from utils.dag_evaluation import DAGEvaluationScaler
    evaluation=DAGEvaluationScaler().load(tmp_path/'model/evaluation_scaler.npz')
    prediction=np.load(tmp_path/'result/test_y_pred.npy')
    truth=np.load(tmp_path/'result/test_y_true.npy')
    windows=np.load(tmp_path/'result/test_window_id.npy')
    report=json.loads((tmp_path/'result/metrics.json').read_text())
    window_mse=[]
    for wid in np.unique(windows):
        valid=(windows==wid)&mask.astype(bool)
        window_mse.append(np.mean(((truth[valid]-prediction[valid])/evaluation.scale_[ids[valid]])**2))
    np.testing.assert_allclose(report['test']['mse_norm'],np.mean(window_mse))
    assert 'mse_norm' in report['best_validation']
    assert report['protocol']['prediction_space']=='original_units'



def test_zero_future_targets_are_not_supervised(tmp_path):
    path, _, _ = archive(tmp_path)
    with np.load(path,allow_pickle=False) as z:
        arrays={key:z[key] for key in z.files}
    arrays['values'][8,0]=0
    arrays['values'][8,1]=0
    np.savez(path,**arrays)
    g={'device':'cpu','exen_data_path':str(path)}
    data=ExEnProcessedProcessor(g,history_len=7,forecast_len=3,target_names=['Y']).process()
    ds=ExEnProcessedDataset(data,type='train',global_args=g,known_future_exogenous_var_ids=[1])
    b=ds.collate_fn([ds[0]])
    assert int(b['batch_pred_mask'].sum())==2
    at_eight=torch.isclose(b['batch_timestamp'],torch.tensor(.8))
    assert not (at_eight & (b['batch_var_idx']==0)).any()
    assert not (at_eight & (b['batch_var_idx']==1)).any()
