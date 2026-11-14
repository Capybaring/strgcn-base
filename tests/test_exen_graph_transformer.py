import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from models import ExEnGraphTransformer
from models.STRGCN.STRGCNDirect import time_features
from models.STRGCN.ExEn import relation_masks
from models.STRGCN.SparseRelations import build_sparse_relations
from models.STRGCN.STRGCN_Layer import masked_softmax
from test_exen import graph
from test_exen_processed import archive

ROOT=Path(__file__).resolve().parents[1]


def model(**kwargs):
    torch.manual_seed(23)
    return ExEnGraphTransformer(num_vars=2,exogenous_var_indices=[0],hid_dim=8,**kwargs)


@pytest.mark.parametrize('cross',[False,True])
def test_augmented_qk_and_layer_match_explicit_graph_attention(cross):
    m=model();layer=m.cross_layers[0] if cross else m.temporal_layers[0]
    b=graph();tt=b['batch_timestamp'];ids=b['batch_var_idx'];types=b['batch_node_type'];valid=b['batch_pad_mask'].bool()
    h=torch.randn(1,7,8);z=layer.norm_attention(h);p=time_features(layer.time_embedding,tt,valid)
    masks=relation_masks(tt,ids,types,valid);message=torch.zeros_like(h)
    for index in range(len(layer.q_proj)):
        q=layer._heads(layer.q_proj[index](z));k=layer._heads(layer.k_proj[index](z));v=layer._heads(layer.v_proj[index](z))
        scores=q@k.transpose(-1,-2)/math.sqrt(layer.head_dim)
        scores=scores+layer.time_strength[index][None,:,None,None]*torch.bmm(p,p.transpose(1,2))[:,None]
        if cross:
            u=layer.adj_U(ids).reshape(1,7,layer.num_heads,-1).transpose(1,2)
            w=layer.adj_V(ids).reshape(1,7,layer.num_heads,-1).transpose(1,2)
            scores=scores+layer.variable_strength[None,:,None,None]*(u@w.transpose(-1,-2))
        aq,ak,_=layer._qkv(z,p,ids,index)
        torch.testing.assert_close(aq@ak.transpose(-1,-2),scores,atol=2e-6,rtol=2e-5)
        mask=masks[2 if cross else index]
        weights=masked_softmax(scores,mask[:,None])
        out=(weights@v).transpose(1,2).reshape(1,7,8)
        out=layer.out_proj[index](out)
        message=message+torch.where(mask.any(-1).unsqueeze(-1),out,0)
    expected=h+message
    expected=expected+layer.ffn(layer.norm_ffn(expected))
    if cross:expected=torch.where(((types==1)|(types==3)).unsqueeze(-1),expected,h)
    plan=build_sparse_relations(tt,ids,types,valid)
    actual=layer(h,None,ids,None,valid,tt,types,2,plan)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('embedding',['rbf','rope','transformer'])
@pytest.mark.parametrize('chunk',[1,3,512])
def test_sdpa_reference_outputs_and_all_parameter_gradients(embedding,chunk):
    args=dict(time_embedding_type=embedding,receiver_chunk_size=chunk)
    a=model(attention_backend='sdpa',**args);c=model(attention_backend='reference',**args)
    c.load_state_dict(a.state_dict())
    b={k:v.repeat(2,1) for k,v in graph().items()};b['batch_pad_mask'][1,0]=0;b['batch_value'][:,4:6]=torch.nan
    x,y=a(**b),c(**b);torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
    x[:,4:6].square().sum().backward();y[:,4:6].square().sum().backward()
    for (name,p),(other,q) in zip(a.named_parameters(),c.named_parameters()):
        assert name==other
        if p.grad is None:assert q.grad is None or (q.grad==0).all(),name
        else:
            assert q.grad is not None,name
            torch.testing.assert_close(p.grad,q.grad,rtol=4e-4,atol=4e-5,msg=name)


@pytest.mark.parametrize('backend',['sdpa','reference'])
def test_target_and_later_future_values_cannot_leak(backend):
    m=model(attention_backend=backend).eval();b=graph();before=m(**b)
    for value in [torch.nan,torch.inf,-1e9]:
        b['batch_value'][:,4:6]=value
        torch.testing.assert_close(m(**b),before)
    b['batch_value'].requires_grad_();m(**b)[:,4:6].sum().backward()
    assert (b['batch_value'].grad[:,4:6]==0).all()
    assert b['batch_value'].grad[0,6].abs()>0
    assert torch.isfinite(m.encoder.weight.grad).all()
    with torch.no_grad():b['batch_value'][0,6]+=100
    after=m(**b);torch.testing.assert_close(after[:,:5],before[:,:5])
    assert not torch.allclose(after[:,5],before[:,5])
    b=graph();b['batch_value'][0,1:4:2]+=100
    torch.testing.assert_close(m(**b)[:,[0,2,6]],before[:,[0,2,6]])


def test_content_changes_attention_at_fixed_metadata():
    m=model();layer=m.cross_layers[0]
    b=graph();ids=b['batch_var_idx'];valid=b['batch_pad_mask'].bool()
    p=time_features(layer.time_embedding,b['batch_timestamp'],valid)
    h=torch.randn(1,7,8);changed=h.clone();changed[0,0,0]+=5
    q,k,_=layer._qkv(layer.norm_attention(h),p,ids,0)
    q2,k2,_=layer._qkv(layer.norm_attention(changed),p,ids,0)
    assert not torch.allclose((q@k.transpose(-1,-2))[0,:,4,0],(q2@k2.transpose(-1,-2))[0,:,4,0])


@pytest.mark.parametrize('no_x',[False,True])
def test_padding_permutation_empty_sources_and_active_ffn(no_x):
    m=model().eval();b=graph()
    if no_x:b={k:v[:,[1,3,4,5]] for k,v in b.items()}
    expected=m(**b);order=torch.arange(b['batch_value'].shape[1]-1,-1,-1)
    torch.testing.assert_close(m(**{k:v[:,order] for k,v in b.items()}),expected[:,order])
    for key,value in b.items():
        fill=-999 if key in ('batch_var_idx','batch_node_type') else float('nan')
        if key in ('batch_pad_mask','batch_pred_mask'):fill=0
        b[key]=torch.cat([value,torch.full((1,2),fill,dtype=value.dtype)],1)
    torch.testing.assert_close(m(**b)[:,:-2],expected)
    assert (m(**b)[:,-2:]==0).all()
    q=b['batch_pred_mask'].bool();m(**b).squeeze(-1)[q].square().sum().backward()
    for layer in list(m.temporal_layers)+list(m.cross_layers):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in layer.ffn.parameters())
    single={k:v[:,4:5] for k,v in graph().items()}
    assert torch.isfinite(m(**single)).all()
    single['batch_pad_mask'].zero_();assert (m(**single)==0).all()


@pytest.mark.parametrize('no_exogenous',[False,True])
def test_cli_both_modes_and_dag_metric_space(tmp_path,no_exogenous):
    path,_,_=archive(tmp_path)
    c=yaml.safe_load((ROOT/'config/ExEnGraphTransformer/processed/BE.yaml').read_text())
    c['global_config']['exen_data_path']=str(path)
    c['data_config']['data_processor_params'].update(history_len=7,forecast_len=3,target_names=['Y'])
    c['train_config']['trainer_params']['max_epoch_num']=1
    cp=tmp_path/'config.yaml';cp.write_text(yaml.safe_dump(c))
    command=[sys.executable,'main.py','--config_path',str(cp),'--device','cpu','--attention-dtype','auto',
             '--model_save_dir_path',str(tmp_path/'model'),'--result_save_dir_path',str(tmp_path/'result')]
    if no_exogenous:command.append('--no-exogenous')
    run=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=60)
    assert run.returncode==0,run.stdout+run.stderr
    report=json.loads((tmp_path/'result/metrics.json').read_text());p=report['protocol']
    assert p['model_name']=='ExEnGraphTransformer' and p['use_exogenous'] is (not no_exogenous)
    assert p['known_future_exogenous_var_ids']==([] if no_exogenous else [1,2])
    assert p['graph_operator']['attention_dtype']=='auto'
    assert p['graph_operator']['total_layers']==8 and p['graph_operator']['bias_implementation']=='augmented_QK'
    from utils.dag_evaluation import DAGEvaluationScaler
    scaler=DAGEvaluationScaler().load(tmp_path/'model/evaluation_scaler.npz')
    mask=np.load(tmp_path/'result/test_y_mask.npy').astype(bool);ids=np.load(tmp_path/'result/test_var_idx.npy')
    actual=np.load(tmp_path/'result/test_y_true.npy');pred=np.load(tmp_path/'result/test_y_pred.npy')
    windows=np.load(tmp_path/'result/test_window_id.npy');assert (ids[mask]==0).all()
    errors=[]
    for wid in np.unique(windows):
        k=(windows==wid)&mask
        errors.append(np.mean(((pred[k]-actual[k])/scaler.scale_[ids[k]])**2))
    np.testing.assert_allclose(report['test']['mse_norm'],np.mean(errors))


def test_configs_match_aligned_tasks_and_training():
    paths=list((ROOT/'config/ExEnGraphTransformer/processed').glob('*.yaml'));assert len(paths)==13
    for path in paths:
        c=yaml.safe_load(path.read_text());old=yaml.safe_load((ROOT/'config/ExEnAligned/processed'/path.name).read_text())
        if path.stem == 'DIST':
            # DIST now shares the t-PatchGNN variable-macro evaluation protocol.
            old['train_config']['trainer_name'] = 'TPatchForecastTrainer'
            old['train_config']['metrics'] = ['mse', 'mae', 'rmse']
            old['train_config']['trainer_params']['eval_metric_key'] = 'mse'
        assert c['data_config']==old['data_config'] and c['train_config']==old['train_config']
        assert c['global_config']==old['global_config']
        p=c['model_config']['model_params'];assert p['n_layer']==4 and p['n_cross_layer']==3
        assert p['hid_dim']==32 and p['num_heads']==4 and p['attention_dtype']=='float32'
        assert p['dropout']==0.3 and p['attention_dropout']==0.0


@pytest.mark.parametrize('no_x',[False,True])
@pytest.mark.parametrize('include_dist',['0','1'])
def test_runners_separate_modes_outputs_and_precision(tmp_path,no_x,include_dist):
    fake=tmp_path/'python';log=tmp_path/'commands.jsonl'
    fake.write_text(f'#!{sys.executable}\nimport json,os,sys\nwith open(os.environ["COMMAND_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env=dict(os.environ,PYTHON=str(fake),DEVICE='cuda:0',ATTENTION_DTYPE='auto',COMMAND_LOG=str(log),
             INCLUDE_DIST=include_dist,NO_EXOGENOUS='0');env.pop('RUN_TAG',None)
    script='run_exen_graph_transformer_no_exogenous.sh' if no_x else 'run_exen_graph_transformer.sh'
    run=subprocess.run(['bash',f'scripts/{script}'],cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
    assert run.returncode==0,run.stderr
    commands=[json.loads(line) for line in log.read_text().splitlines()];assert len(commands)==12+int(include_dist)
    tag='exen_graph_transformer_no_exogenous' if no_x else 'exen_graph_transformer_known_future'
    for command in commands:
        p=command[command.index('--config_path')+1];name=Path(p).stem
        assert p==f'config/ExEnGraphTransformer/processed/{name}.yaml'
        assert ('--no-exogenous' in command)==no_x
        assert command[command.index('--attention-dtype')+1]=='auto'
        effective_tag = tag + '_rolling_cfg' if name == 'DIST' else tag
        assert f'results/{effective_tag}/{name}' in command
