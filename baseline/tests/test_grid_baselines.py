import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from baseline.timexer_irregular.adapter import IrregularTimeXer
from baseline.crosslinear_irregular.adapter import IrregularCrossLinear
from baseline.common.run import run,make_parser

CLASSES={'timexer':IrregularTimeXer,'crosslinear':IrregularCrossLinear}


def model(method,exo=True,targets=(5,2)):
    torch.manual_seed(9)
    params=dict(seq_len=8,patch_len=4,d_model=8,d_ff=16,use_exogenous=exo)
    if method=='timexer':params.update(n_heads=2,e_layers=1,dropout=0.)
    else:params['fusion_hidden']=8
    return CLASSES[method](list(targets),[9],7.,3.,**params)


def batch():
    return dict(batch_value=torch.tensor([[2.,4.,3.,6.,10.,20.,30.,999.,888.,float('nan')]]),
        batch_timestamp=torch.tensor([[0.,.6,0.,.6,0.,.6,.8,.99,.7,float('nan')]],dtype=torch.float64),
        batch_var_idx=torch.tensor([[5,5,2,2,9,9,9,5,2,5]]),
        batch_node_type=torch.tensor([[1,1,1,1,0,0,2,3,3,3]]),
        batch_pred_mask=torch.tensor([[0.,0.,0.,0.,0.,0.,0.,1.,1.,0.]]),
        batch_pad_mask=torch.tensor([[1.,1.,1.,1.,1.,1.,1.,1.,1.,0.]]))


@pytest.mark.parametrize('method',CLASSES)
@pytest.mark.parametrize('exo',[True,False])
def test_query_gradients_label_isolation_and_x_availability(method,exo):
    m=model(method,exo);b=batch()
    out=m(**b)
    assert out.shape==(1,10,1) and torch.isfinite(out).all()
    loss=out[0,7:9].square().mean();loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.core.model.parameters())
    if method=='crosslinear' and exo:
        assert all(p.grad is not None for p in m.core.future_fusion.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval();expected=m(**b)
    b['batch_value'][0,7:]=float('nan')
    torch.testing.assert_close(expected,m(**b),rtol=0,atol=0)
    b['batch_value'][0,4:7]=torch.tensor([-100.,200.,300.])
    changed=m(**b)
    if exo:assert not torch.allclose(expected,changed)
    else:torch.testing.assert_close(expected,changed,rtol=0,atol=0)
    b['batch_pad_mask'][0,6]=0
    expected=m(**b);b['batch_value'][0,6]=1e10
    torch.testing.assert_close(expected,m(**b),rtol=0,atol=0)
    b['batch_timestamp'][0,7]=1.01
    with pytest.raises(ValueError,match='queries'):m(**b)


@pytest.mark.parametrize('method',CLASSES)
def test_source_provenance_and_relocation(method):
    root=Path(__file__).parents[1]/f'{method}_irregular/vendor'
    meta=json.loads((root/'PROVENANCE.json').read_text())
    for name,info in meta['files'].items():
        raw=(root/name).read_bytes()
        if meta['import_replacements']:
            original=raw.decode()
            for old,new in sorted(meta['import_replacements'].items(),key=lambda pair:len(pair[1]),reverse=True):
                original=original.replace(new,old)
            raw=original.encode()
        assert hashlib.sha256(raw).hexdigest()==info['sha256']


def test_crosslinear_target_order_and_future_fusion_contract():
    m=model('crosslinear').eval();b=batch();h,f=m.grids(**b)
    class IdentityForecast(torch.nn.Module):
        def forward(self,x,*args):return x[:,-m.pred_len:,:]
    m.core.model=IdentityForecast();m.core.future_fusion=None
    out,_=m.core(h,f)
    torch.testing.assert_close(out,h[:,-m.pred_len:,:2])


@pytest.mark.parametrize('method',CLASSES)
@pytest.mark.parametrize('mode',['FullExo','NoExo'])
@pytest.mark.parametrize('multi',[True,False])
def test_training_checkpoint_metrics_and_exact_query_exports(tmp_path,method,mode,multi):
    import yaml
    time=1700000000.+np.arange(181)*.1731234567
    values=np.column_stack([np.sin(np.arange(181))+3,np.arange(181)+1,np.cos(np.arange(181))+4]).astype('float32')
    mask=np.ones_like(values,dtype=bool);mask[::3,0]=False;mask[::5,1]=False;mask[::4,2]=False
    data=tmp_path/'stream.npz';np.savez(data,time=time,values=values,mask=mask,names=['Y1','X','Y2'])
    targets=['Y2','Y1'] if multi else ['Y1']
    config=dict(global_config=dict(exen_data_path=str(data)),data_config=dict(
        data_processor_params=dict(history_len=1.4,forecast_len=.6,target_names=targets),
        dataset_params=dict(known_future_exogenous_var_ids='all')))
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    out=tmp_path/'run'
    args=make_parser(method).parse_args(['--config',str(path),'--mode',mode,'--output',str(out),
        '--tiny','--epochs','2','--max-windows','3','--batch-size','2'])
    result=run(args)
    assert result['test']['forecast_window_count']==3
    assert np.isfinite(result['test']['mse_norm'])
    assert (out/'model.pth').exists() and len(result['training_history'])==2
    with np.load(out/'predictions.npz') as z:
        assert all(t in time for t in z['timestamp'])
        assert len(z['prediction'])==sum(result['test']['observed_count_by_target'])
        assert set(z['variable_id'])==({2,0} if multi else {0})
        assert np.isfinite(z['prediction']).all()
        # Reproduce checkpoint selection in input units (equal batch weight).
        with np.load(out/'input_scaler.npz') as scale:
            e=np.abs(z['prediction']-z['actual'])/scale['ranges'][z['variable_id']]
        e=e**2 if args.loss=='mse' else np.where(e<=.5,.5*e**2,.5*(e-.25))
        groups=z['window_id']//2
        expected=np.mean([e[groups==i].mean() for i in np.unique(groups)])
        assert result['test']['loss']==pytest.approx(expected)
    with pytest.raises(FileExistsError):run(args)


@pytest.mark.parametrize('method',CLASSES)
def test_empty_and_singleton_sources_are_finite(method):
    m=model(method);b=batch();b['batch_pad_mask'][0,:7]=0
    out=m(**b)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    b['batch_pad_mask'][0,0]=1
    h,f=m.grids(**b)
    assert (h[:,:,0]==2).all() and (h[:,:,1:]==0).all() and (f==0).all()
    assert torch.isfinite(m(**b)).all()


def test_full_exogenous_batch_script_covers_every_dataset(tmp_path):
    import os
    import subprocess
    repo=Path(__file__).resolve().parents[2]
    recorder=tmp_path/'python'
    recorder.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$INVOCATIONS"\n')
    recorder.chmod(0o755)
    log=tmp_path/'calls'
    env=dict(os.environ,PYTHON=str(recorder),INVOCATIONS=str(log),RUN_ROOT=str(tmp_path/'runs'),
             DEVICE='cpu',EPOCHS='2',BATCH_SIZE='2')
    subprocess.run(['bash',str(repo/'baseline/run_full_exo.sh')],env=env,cwd=tmp_path,check=True,capture_output=True)
    lines=log.read_text().splitlines()
    configs=sorted((repo/'config/ExEn/processed').glob('*.yaml'))
    assert len(lines)==2*len(configs)
    for method in CLASSES:
        for config in configs:
            matched=[line for line in lines if f'baseline.{method}_irregular.run' in line and f'processed/{config.name} ' in line]
            assert len(matched)==1 and '--mode FullExo' in matched[0]
            assert '--device cpu' in matched[0]
            assert '--tiny' not in matched[0] and '--max-windows' not in matched[0]
