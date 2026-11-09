"""DAG MLP structure, ragged adapters, information isolation and CLI runs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from baseline.patch_fusion.adapter import DAGMLPFusion, TPatchGNNFusion, HiPatchFusion
from baseline.patch_fusion.vendor.compat import MessagePassing, scatter
from data_processors import TPatchRecordProcessor
from datasets import TPatchFusionDataset
from utils.scaler import TPatchInputScaler
from test_tpatch_records import make_data

ROOT = Path(__file__).resolve().parents[1]


def prepared(root, dataset='activity', mode='FullExo'):
    g = {'device':'cpu'}
    data = TPatchRecordProcessor(g,dataset,data_root=root,max_samples_per_split=2).process()
    scale = TPatchInputScaler(); scale.fit(data['seen_data'])
    ps = dict(mimic=8,physionet=8,activity=300,ushcn=2)[dataset]
    ds = TPatchFusionDataset(scale.transform(data['train_data']),type='train',global_args=g,
        patch_size=ps,use_exogenous=mode!='NoExo',
        known_future_exogenous_var_ids=[] if mode=='HistExo' else 'all')
    return g,data,ds


def test_fusion_is_exact_two_linear_relu_without_residual():
    torch.manual_seed(2);m = DAGMLPFusion(3,2,64)
    x,z = torch.randn(2,4,2),torch.randn(2,4,3)
    expected=F.linear(F.relu(F.linear(torch.cat([z,x],-1),m.linear1.weight,m.linear1.bias)),m.linear2.weight,m.linear2.bias)
    torch.testing.assert_close(m(x,z),expected)
    assert sum(isinstance(c,torch.nn.Linear) for c in m.modules())==2
    for p in m.parameters():p.data.zero_()
    assert (m(x,z)==0).all()  # a residual adapter would return z instead


@pytest.mark.parametrize('cls',[TPatchGNNFusion,HiPatchFusion])
@pytest.mark.parametrize('dataset',['mimic','physionet','activity','ushcn'])
@pytest.mark.parametrize('mode',['FullExo','HistExo','NoExo'])
def test_backbones_forward_backward_and_input_roles(tmp_path,cls,dataset,mode):
    make_data(tmp_path,dataset)
    g,_,ds=prepared(tmp_path,dataset,mode)
    b=ds.collate_fn([ds[0],ds[1]])
    m=cls(global_args=g,hid_dim=8,n_layer=1,n_heads=1).eval()
    pred=m(**b).squeeze(-1);q=b['batch_pred_mask'].bool()
    assert pred.shape==b['batch_value'].shape and torch.isfinite(pred).all()
    loss=(pred[q]-b['batch_value'][q]).square().mean();loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.core.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.fusion.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    if mode in ['HistExo','NoExo']:
        assert (b['batch_backbone_future_x']==0).all()
    if mode=='NoExo':
        assert (b['batch_backbone_mask'][...,len(ds.packet['target_ids']):]==0).all()


@pytest.mark.parametrize('cls',[TPatchGNNFusion,HiPatchFusion])
def test_future_y_never_enters_patches_and_later_x_cannot_change_earlier_query(tmp_path,cls):
    make_data(tmp_path,'activity')
    g,_,ds=prepared(tmp_path)
    first=ds[0];sample,origin=first
    batch=ds.collate_fn([first])
    m=cls(global_args=g,hid_dim=8,n_layer=1,n_heads=1).eval()
    before=m(**batch)
    q=sample[:,3].bool() & torch.isin(sample[:,2].long(),torch.tensor(ds.packet['target_ids']))
    altered=sample.clone();altered[q,1]=1e8
    modified=ds.collate_fn([(altered,origin)])
    for key in ('batch_backbone_values','batch_backbone_mask','batch_backbone_times','batch_backbone_future_x'):
        torch.testing.assert_close(batch[key],modified[key])
    torch.testing.assert_close(before,m(**modified))
    # Perturb the last supplied future X after at least one earlier Y query.
    role=batch['batch_node_type'][0];t=batch['batch_timestamp'][0]
    x=torch.where((role==2)&batch['batch_pad_mask'][0].bool())[0]
    later=x[t[x].argmax()];cut=t[later]
    altered=sample.clone();altered[later,1]+=100
    changed=ds.collate_fn([(altered,origin)])
    earlier=batch['batch_pred_mask'].bool()&(batch['batch_timestamp']<cut)
    assert earlier.any()
    torch.testing.assert_close(before[earlier],m(**changed)[earlier])


def test_causal_fill_keeps_zero_and_never_backfills(tmp_path):
    make_data(tmp_path,'activity')
    g,_,ds=prepared(tmp_path)
    sample,origin=ds[0]
    # Only exogenous column 6 at times .8 and .9; earlier query has no prior X.
    sample=sample[~(sample[:,2]==6)]
    extra=torch.tensor([[.8,0.,6.,1.],[.9,123.,6.,1.]])
    sample=torch.cat([sample,extra]);sample=sample[torch.argsort(sample[:,0],stable=True)]
    b=ds.collate_fn([(sample,origin)])
    q=b['batch_backbone_queries'][0]
    values=b['batch_backbone_future_x'][0,:,0]
    assert (values[q<.9]==0).all()
    assert (values[q>=.9]==123).all()


def test_torch_fallback_sum_aggregation_values_and_gradients():
    class Layer(MessagePassing):
        def message(self,x_j,x_i,edge_index_i,*args):return x_j*2+x_i
        def update(self,aggr_out,residual):return aggr_out+residual
    x=torch.randn(4,3,requires_grad=True);edges=torch.tensor([[0,1,2,2],[1,1,0,3]])
    none=torch.zeros(4,1)
    out=Layer().propagate(edges,x,none,none,none,none,0,x)
    expected=x.clone()
    for src,dst in edges.T:expected[dst]=expected[dst]+2*x[src]+x[dst]
    torch.testing.assert_close(out,expected)
    a=torch.autograd.grad(out.square().sum(),x,retain_graph=True)[0]
    b=torch.autograd.grad(expected.square().sum(),x)[0]
    torch.testing.assert_close(a,b)
    src=torch.randn(4,2)
    expected=torch.stack([src[2],src[0]+src[1],torch.zeros(2),src[3]])
    torch.testing.assert_close(scatter(src,edges[1],dim_size=4),expected)


@pytest.mark.parametrize('model',['TPatchGNNFusion','HiPatchFusion'])
@pytest.mark.parametrize('dataset',['mimic','physionet','activity','ushcn'])
def test_cli_shared_tpatch_metrics_and_source_exports(tmp_path,model,dataset):
    make_data(tmp_path,dataset)
    c=yaml.safe_load((ROOT/f'config/{model}/tpatch/{dataset}.yaml').read_text())
    c['data_config']['data_processor_params']['max_samples_per_split']=1
    c['model_config']['model_params'].update(hid_dim=8,n_layer=1,n_heads=1)
    c['train_config']['trainer_params']['max_epoch_num']=1
    config=tmp_path/'run.yaml';config.write_text(yaml.safe_dump(c))
    args=[sys.executable,'main.py','--config_path',str(config),'--device','cpu',
          '--tpatch-data-root',str(tmp_path),'--model_save_dir_path',str(tmp_path/'model'),
          '--result_save_dir_path',str(tmp_path/'result')]
    if dataset=='physionet':args+=['--history-only-exogenous']
    if dataset=='activity':args+=['--no-exogenous']
    run=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=90)
    assert run.returncode==0,run.stdout+run.stderr
    report=json.loads((tmp_path/'result/metrics.json').read_text())
    assert np.isfinite(report['test']['mse']) and 'mse_norm' not in report['test']
    assert report['protocol']['fusion']['module']=='DAG two-layer pointwise MLP, no residual'
    assert report['protocol']['model_name']==model
    for name in ['test_y_mask.npy','test_var_idx.npy','test_record_id.npy','test_window_id.npy']:
        assert (tmp_path/'result'/name).exists()


def test_runner_separate_modes_and_model_outputs(tmp_path):
    fake=tmp_path/'python';log=tmp_path/'calls'
    fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CALLS"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    for mode,flag in [('FullExo',None),('HistExo','--history-only-exogenous'),('NoExo','--no-exogenous')]:
        run=subprocess.run(['bash','scripts/run_patch_fusion.sh','activity'],cwd=ROOT,
            env=dict(os.environ,PYTHON=str(fake),CALLS=str(log),MODE=mode),capture_output=True,text=True)
        assert run.returncode==0,run.stderr
    calls=[json.loads(x) for x in log.read_text().splitlines()]
    assert len(calls)==6
    for i,args in enumerate(calls):
        expected=('--history-only-exogenous' if i in (2,3) else '--no-exogenous' if i in (4,5) else None)
        if expected:assert expected in args
    outputs=[a[a.index('--result_save_dir_path')+1] for a in calls]
    assert len(set(outputs))==6


def test_packed_history_preserves_each_historical_observation(tmp_path):
    make_data(tmp_path,'activity')
    _,_,ds=prepared(tmp_path)
    samples=[ds[0],ds[1]];b=ds.collate_fn(samples)
    for i,(sample,origin) in enumerate(samples):
        historical=b['batch_pad_mask'][i].bool() & (b['batch_node_type'][i]<2)
        assert int(b['batch_backbone_mask'][i].sum())==int(historical.sum())
        for original,local in ds.local_id.items():
            take=historical & (b['batch_var_idx'][i]==original)
            patchmask=b['batch_backbone_mask'][i,...,local].bool()
            expected=b['batch_value'][i,take].sort().values
            actual=b['batch_backbone_values'][i,...,local][patchmask].sort().values
            torch.testing.assert_close(expected,actual)


@pytest.mark.parametrize('cls',[TPatchGNNFusion,HiPatchFusion])
def test_no_historical_observation_stays_finite(tmp_path,cls):
    make_data(tmp_path,'activity')
    g,_,ds=prepared(tmp_path,mode='NoExo')
    sample,origin=ds[0]
    sample=sample[sample[:,3].bool()]
    b=ds.collate_fn([(sample,origin)])
    assert not b['batch_backbone_mask'].any()
    m=cls(global_args=g,hid_dim=8,n_layer=1,n_heads=1)
    assert torch.isfinite(m(**b)).all()
