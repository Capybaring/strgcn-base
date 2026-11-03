import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from models import ExEnAligned
from models.STRGCN.ExEn import relation_masks
from models.STRGCN.ExEnAligned import AlignedRelationLayer
from models.STRGCN.SparseRelations import build_sparse_relations
from models.STRGCN.STRGCNDirect import time_features
from models.STRGCN.STRGCN_Layer import low_rank_cheb_graph_conv, masked_softmax
from test_exen import graph
from test_exen_processed import archive

ROOT = Path(__file__).resolve().parents[1]


def model(**kwargs):
    torch.manual_seed(17)
    return ExEnAligned(num_vars=2,exogenous_var_indices=[0],hid_dim=8,receiver_chunk_size=2,**kwargs)


@pytest.mark.parametrize('backend',['dense','sparse_csr'])
@pytest.mark.parametrize('cross',[False,True])
@pytest.mark.parametrize('depth',[1,2,3])
def test_aligned_layer_matches_original_receiver_cheb(backend,cross,depth):
    torch.manual_seed(42)
    layer = AlignedRelationLayer(8,4,2,0.,True,True,cross=cross,depth=depth)
    b = graph();ids=b['batch_var_idx'];tt=b['batch_timestamp'];types=b['batch_node_type']
    valid=b['batch_pad_mask'].bool();exo=(types==0)|(types==2)
    masks=relation_masks(tt,ids,types,valid)
    selected=masks[2:] if cross else masks[:2]
    plan=build_sparse_relations(tt,ids,types,valid) if backend=='sparse_csr' else None
    h=torch.randn(1,7,8);p=time_features(layer.time_embedding,tt,valid)
    message=torch.zeros_like(h)
    for index,mask in enumerate(selected):
        logits=layer.time_scale[index]*layer.time_temp[index]*torch.bmm(p,p.transpose(1,2))
        if cross:
            logits=logits+layer.var_scale*layer.var_temp*torch.bmm(layer.adj_U(ids),layer.adj_V(ids).transpose(1,2))
        a=masked_softmax(logits,mask)
        ws=torch.stack([embedding(ids[0]) for embedding in layer.weight_S[index]])
        branch=low_rank_cheb_graph_conv(h,a,layer.weight_T[index](p),ws,depth)
        domain=valid & (~exo if cross or index==1 else exo)
        message=message+torch.where(domain.unsqueeze(-1),branch,0)
    expected=layer.norm(layer.lambda_*(message+layer.bias).relu()+(1-layer.lambda_)*h)
    if cross:expected=torch.where((valid & ~exo).unsqueeze(-1),expected,h)
    actual=layer(h,None,ids,selected,valid,tt,types,2,plan)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('embedding',['rbf','rope','transformer'])
@pytest.mark.parametrize('depth',[2,3])
def test_dense_csr_predictions_and_parameter_gradients(embedding,depth):
    args=dict(time_embedding_type=embedding,gcn_depth=depth)
    dense=model(relation_backend='dense',**args)
    sparse=model(relation_backend='sparse_csr',**args)
    sparse.load_state_dict(dense.state_dict())
    b={k:v.repeat(2,1) for k,v in graph().items()}
    b['batch_pad_mask'][1,0]=0
    b['batch_value'][:,4:6]=torch.nan
    a,c=dense(**b),sparse(**b)
    torch.testing.assert_close(a,c,rtol=3e-5,atol=3e-6)
    a[:,4:6].square().sum().backward();c[:,4:6].square().sum().backward()
    for (name,p),(other,q) in zip(dense.named_parameters(),sparse.named_parameters()):
        assert name==other
        if p.grad is None:
            assert q.grad is None or (q.grad==0).all(),name
        else:
            assert q.grad is not None,name
            torch.testing.assert_close(p.grad,q.grad,rtol=6e-4,atol=6e-5,msg=name)


@pytest.mark.parametrize('backend',['dense','sparse_csr'])
def test_no_label_or_late_future_leak_and_no_y_to_x_path(backend):
    m=model(relation_backend=backend,gcn_depth=3).eval();b=graph()
    expected=m(**b)
    for value in (torch.nan,torch.inf,-1e9):
        b['batch_value'][:,4:6]=value
        torch.testing.assert_close(m(**b),expected)
    b['batch_value'].requires_grad_();m(**b)[:,4:6].square().sum().backward()
    assert (b['batch_value'].grad[:,4:6]==0).all()
    assert b['batch_value'].grad[0,6].abs()>0
    assert torch.isfinite(m.encoder.weight.grad).all()
    with torch.no_grad():
        b['batch_value'][0,6]+=100
        changed=m(**b)
    torch.testing.assert_close(changed[:,:5],expected[:,:5])
    assert not torch.allclose(changed[:,5],expected[:,5])
    b=graph();b['batch_value'][0,1:4:2]+=100
    # Y history never enters the X states, through any stage.
    torch.testing.assert_close(m(**b)[:,[0,2,6]],expected[:,[0,2,6]])


@pytest.mark.parametrize('backend',['dense','sparse_csr'])
def test_y_only_keeps_cheb_self_update_in_all_three_layers(backend):
    m=model(relation_backend=backend)
    # Retain exactly historical Y and query Y.
    b={k:v[:,[1,3,4,5]] for k,v in graph().items()}
    b['batch_value'].requires_grad_();q=b['batch_pred_mask'].bool()
    out=m(**b).squeeze(-1)
    loss=(out[q]-b['batch_value'].detach()[q]).square().mean();loss.backward()
    assert torch.isfinite(loss) and (b['batch_value'].grad[q]==0).all()
    layers=list(m.temporal_layers)+list(m.cross_layers)
    assert len(layers)==3 and len({id(layer.time_embedding) for layer in layers})==3
    for layer in layers:
        assert layer.depth==2
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in layer.parameters())
    # In particular, the cross T0 factor remains trainable when X is absent.
    assert m.cross_layers[0].weight_S[0][0].weight.grad.abs().sum()>0


@pytest.mark.parametrize('backend',['dense','sparse_csr'])
def test_padding_empty_query_and_node_order(backend):
    m=model(relation_backend=backend).eval();b=graph();expected=m(**b)
    order=torch.tensor([6,3,5,0,2,4,1])
    torch.testing.assert_close(m(**{k:v[:,order] for k,v in b.items()}),expected[:,order])
    for key,value in b.items():
        fill=-999 if key in ('batch_var_idx','batch_node_type') else float('nan')
        if key in ('batch_pad_mask','batch_pred_mask'):fill=0
        b[key]=torch.cat([value,torch.full((1,2),fill,dtype=value.dtype)],1)
    result=m(**b)
    torch.testing.assert_close(result[:,:7],expected)
    assert (result[:,7:]==0).all()
    single={k:v[:,4:5] for k,v in graph().items()}
    assert torch.isfinite(m(**single)).all()
    single['batch_pad_mask'].zero_()
    assert (m(**single)==0).all()


@pytest.mark.parametrize('no_exogenous',[False,True])
def test_full_cli_training_evaluation_and_operator_manifest(tmp_path,no_exogenous):
    path,_,_=archive(tmp_path)
    c=yaml.safe_load((ROOT/'config/ExEnAligned/processed/BE.yaml').read_text())
    c['global_config']['exen_data_path']=str(path)
    c['data_config']['data_processor_params'].update(history_len=7,forecast_len=3,target_names=['Y'])
    c['train_config']['trainer_params']['max_epoch_num']=1
    cp=tmp_path/'config.yaml';cp.write_text(yaml.safe_dump(c))
    command=[sys.executable,'main.py','--config_path',str(cp),'--device','cpu',
             '--model_save_dir_path',str(tmp_path/'model'),'--result_save_dir_path',str(tmp_path/'result')]
    if no_exogenous:command.append('--no-exogenous')
    run=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=60)
    assert run.returncode==0,run.stdout+run.stderr
    report=json.loads((tmp_path/'result/metrics.json').read_text());pro=report['protocol']
    assert pro['model_name']=='ExEnAligned' and pro['use_exogenous'] is (not no_exogenous)
    assert pro['relation_scope']=='role_constrained_exen_staged_aligned'
    assert pro['known_future_exogenous_var_ids']==([] if no_exogenous else [1,2])
    assert pro['graph_operator']==dict(gcn_type='cheb',chebyshev_terms=2,variable_logit_normalizer=1.0,
        factorization='receiver_after_propagation',layer_time_features='independent',total_layers=3,
        empty_neighborhood='keep_T0_self_update')
    from utils.dag_evaluation import DAGEvaluationScaler
    scaler=DAGEvaluationScaler().load(tmp_path/'model/evaluation_scaler.npz')
    mask=np.load(tmp_path/'result/test_y_mask.npy').astype(bool)
    ids=np.load(tmp_path/'result/test_var_idx.npy')
    pred=np.load(tmp_path/'result/test_y_pred.npy');true=np.load(tmp_path/'result/test_y_true.npy')
    windows=np.load(tmp_path/'result/test_window_id.npy')
    assert mask.any() and (ids[mask]==0).all()
    errors=[]
    for wid in np.unique(windows):
        selected=(windows==wid)&mask
        errors.append(np.mean(((pred[selected]-true[selected])/scaler.scale_[ids[selected]])**2))
    np.testing.assert_allclose(report['test']['mse_norm'],np.mean(errors))


def test_all_configs_preserve_data_training_and_shared_operator_settings():
    paths=list((ROOT/'config/ExEnAligned/processed').glob('*.yaml'))
    assert len(paths)==13
    for path in paths:
        c=yaml.safe_load(path.read_text())
        joint=yaml.safe_load((ROOT/'config/STRGCNDirect/processed'/path.name).read_text())
        assert c['data_config']==joint['data_config'] and c['train_config']==joint['train_config']
        assert c['global_config']==joint['global_config']
        params=c['model_config']['model_params']
        assert c['model_config']['model_name']=='ExEnAligned'
        assert params['n_layer']==2 and params['n_cross_layer']==1
        for key in set(joint['model_config']['model_params'])-{'n_layer'}:
            assert params[key]==joint['model_config']['model_params'][key],(path,key)


@pytest.mark.parametrize('mode',['future','no_x','wrapper'])
@pytest.mark.parametrize('include_dist',['0','1'])
def test_batch_runner_modes_and_separate_outputs(tmp_path,mode,include_dist):
    fake=tmp_path/'python'
    fake.write_text(f'#!{sys.executable}\nimport json,os,sys\nwith open(os.environ["COMMAND_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755);log=tmp_path/'commands.jsonl'
    env=dict(os.environ,PYTHON=str(fake),DEVICE='cuda:0',COMMAND_LOG=str(log),
             INCLUDE_DIST=include_dist,NO_EXOGENOUS='1' if mode=='no_x' else '0')
    env.pop('RUN_TAG',None)
    script='run_exen_aligned_no_exogenous.sh' if mode=='wrapper' else 'run_exen_aligned.sh'
    run=subprocess.run(['bash',f'scripts/{script}'],cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
    assert run.returncode==0,run.stderr
    commands=[json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands)==12+int(include_dist)
    tag='exen_aligned_known_future' if mode=='future' else 'exen_aligned_no_exogenous'
    names=[]
    for command in commands:
        config=command[command.index('--config_path')+1];name=Path(config).stem;names.append(name)
        assert config==f'config/ExEnAligned/processed/{name}.yaml'
        assert ('--no-exogenous' in command)==(mode!='future')
        assert command[command.index('--device')+1]=='cuda:0'
        assert f'results/{tag}/{name}' in command and f'model_states/{tag}/{name}' in command
    assert ('DIST' in names)==(include_dist=='1')
