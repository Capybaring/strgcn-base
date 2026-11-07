import hashlib
import json
from pathlib import Path
import pytest
import torch
from .adapter import interpolate, IrregularDAG


def model(exo=True):
    torch.manual_seed(4)
    return IrregularDAG([0], [1], 7., 3., seq_len=8, use_exogenous=exo,
        patch_len=4, stride=2, d_model=8, d_ff=16, n_heads=2, e_layers=1, dropout=0.)


def batch():
    return dict(batch_value=torch.tensor([[2.,4.,10.,20.,30.,999.,float('nan')]]),
        batch_timestamp=torch.tensor([[0.,.6,0.,.6,.8,.99,float('nan')]]),
        batch_var_idx=torch.tensor([[0,0,1,1,1,0,0]]),
        batch_node_type=torch.tensor([[1,1,0,0,2,3,3]]),
        batch_pad_mask=torch.tensor([[1.,1.,1.,1.,1.,1.,0.]]),
        batch_pred_mask=torch.tensor([[0.,0.,0.,0.,0.,1.,0.]]))


def test_interpolation_boundaries_duplicates_and_gradients():
    t = torch.tensor([2.,0.,2.,4.])
    y = torch.tensor([2.,0.,6.,8.], requires_grad=True)
    q = torch.tensor([-1.,0.,1.,2.,3.,4.,5.])
    out = interpolate(t,y,q)
    torch.testing.assert_close(out,torch.tensor([0.,0.,2.,4.,6.,8.,8.]))
    out.sum().backward()
    assert y.grad.isfinite().all() and (y.grad > 0).all()
    torch.testing.assert_close(interpolate(t[:0],y[:0],q),torch.zeros(7))
    torch.testing.assert_close(interpolate(t[:1],y[:1],q),torch.full((7,),2.))
    with pytest.raises(ValueError):interpolate(t,y,torch.tensor([float('nan')]))


def test_history_ignores_future_labels_and_future_x():
    m=model();b=batch();history,future=m.grids(**b)
    b['batch_value'][0,5]=-1e9;b['batch_value'][0,4]=300.
    h2,f2=m.grids(**b)
    torch.testing.assert_close(history,h2)
    assert not torch.equal(future,f2)
    assert torch.isfinite(history).all() and torch.isfinite(future).all()


@pytest.mark.parametrize('exo',[True,False])
def test_forward_backward_no_label_leakage(exo):
    m=model(exo);b=batch();out=m(**b)
    loss=(out[0,5,0]-5).abs()+m.additional_loss;loss.backward()
    for branch in (m.core.temporal_encoder,m.core.cov_encoder):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.parameters())
    assert torch.isfinite(loss)
    m.eval();expected=m(**b);b['batch_value'][0,5]=float('nan')
    torch.testing.assert_close(expected,m(**b))
    if not exo:
        assert m.additional_loss==0


def test_strict_noexo_with_injected_x_and_padding():
    m=model(False).eval();b=batch();expected=m(**b)
    b['batch_value'][0,2:5]=torch.tensor([float('nan'),-1e10,1e10])
    b['batch_timestamp'][0,2:5]=torch.tensor([.123,.321,.999])
    b['batch_pad_mask'][0,2:5]=0
    torch.testing.assert_close(expected,m(**b),rtol=0,atol=0)
    h,f=m.grids(**b)
    assert (h[:,:,-1]==0).all() and (f==0).all()


def test_output_time_semantics_and_edge_gradient():
    m=model();b=batch();pred_len=len(m.future_grid)
    class LinearCore(torch.nn.Module):
        def forward(self,h,f):
            self.values=torch.arange(pred_len,dtype=h.dtype)[None,:,None].requires_grad_()
            return self.values,0
    core=LinearCore();m.core=core;out=m(**b)
    expected=(.99-m.boundary)/(m.boundary/8)
    assert out[0,5,0].item()==pytest.approx(expected,abs=1e-5)
    out.sum().backward()
    assert (core.values.grad[0,-2:,0]>0).all()
    assert m.future_grid[-1]>=1
    b['batch_timestamp'][0,5]=1.01
    with pytest.raises(ValueError):m(**b)


def test_missing_channel_and_singleton_history_hold():
    m=model();b=batch();b['batch_pad_mask'][0,2:5]=0
    h,f=m.grids(**b)
    assert (h[:,:,-1]==0).all() and (f==0).all()
    b['batch_pad_mask'][0,2]=1
    h,f=m.grids(**b)
    assert (h[:,:,-1]==10).all() and (f==10).all()


def test_vendored_computation_matches_original_hashes():
    vendor=Path(__file__).parent/'vendor'
    provenance=json.loads((vendor/'PROVENANCE.json').read_text())
    for source,digest in provenance['sha256'].items():
        original=(vendor/Path(source).name).read_text()
        for module in ('Embed','CC_EncDec','TC_EncDec','SelfAttention_Family','Transformer_EncDec'):
            original=original.replace('from .'+module,'from ts_benchmark.baselines.dag.layers.'+module)
        original=original.replace('from .masking','from ts_benchmark.baselines.time_series_library.utils.masking')
        if Path(source).name == 'CC_EncDec.py':
            original += '\n'  # Restore upstream trailing blank line for verification.
        assert hashlib.sha256(original.encode()).hexdigest()==digest


@pytest.mark.parametrize('mode',['FullExo','NoExo'])
def test_end_to_end_training_scaling_metrics_and_exact_times(tmp_path, mode):
    import argparse
    import numpy as np
    import yaml
    from .run import run
    from .dataset import DAGProcessedDataset
    from data_processors.ExEnProcessedProcessor import ExEnProcessedProcessor
    from utils.scaler import ExEnProcessedMinMaxScaler
    time=1700000000.+np.arange(121)*.1731234567
    values=np.column_stack([np.sin(np.arange(121)) + 3,np.arange(121)+1]).astype('float32')
    mask=np.ones_like(values,dtype=bool);mask[::3,0]=False;mask[::5,1]=False
    data=tmp_path/'irregular.npz'
    np.savez(data,time=time,values=values,mask=mask,names=['Y','X'])
    config=dict(global_config=dict(exen_data_path=str(data)),data_config=dict(
        data_processor_params=dict(history_len=1.4,forecast_len=.6,target_names=['Y']),
        dataset_params=dict(known_future_exogenous_var_ids=[])))
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    out=tmp_path/mode
    result=run(argparse.Namespace(config=str(path),data=None,seed=4,threads=1,device='cpu',
        max_windows=2,mode=mode,tiny=True,output=str(out),lr=.002,epochs=2,batch_size=2))
    assert result['test']['forecast_window_count']==2
    assert np.isfinite(result['test']['mse_norm'])
    with np.load(out/'predictions.npz') as z:
        assert all(v in time for v in z['timestamp'])
        assert (z['variable_id']==0).all()
        assert len(z['prediction'])==sum(result['test']['observed_count_by_target'])
        assert np.isfinite(z['prediction']).all()
    # Same splits, train-only statistics, exact times, and unknown future X removed.
    packets=ExEnProcessedProcessor(dict(exen_data_path=str(data)),**config['data_config']['data_processor_params'],
                                  max_windows_per_split=2).process()
    scaler=ExEnProcessedMinMaxScaler();scaler.fit(packets['train_data'])
    ds=DAGProcessedDataset(scaler.transform(packets['train_data']),type='train',global_args={},
                          known_future_exogenous_var_ids=[],use_exogenous=mode=='FullExo')
    b=ds.collate_fn([ds[0]])
    assert b['batch_timestamp'].dtype==torch.float64
    assert not (b['batch_node_type']==2).any()
    if mode=='NoExo':assert not (b['batch_var_idx']==1).any()
