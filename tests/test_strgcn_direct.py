"""Independent checks against original operators, leakage and the full CLI."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from models import STRGCNDirect
from models.STRGCN.STRGCNDirect import DirectSTRGCNLayer, joint_sparse_plan, time_features
from models.STRGCN.STRGCN_Layer import low_rank_cheb_graph_conv, low_rank_graph_conv, masked_softmax
from test_exen import graph
from test_exen_processed import archive

ROOT = Path(__file__).resolve().parents[1]


def direct(**kwargs):
    torch.manual_seed(123)
    return STRGCNDirect(num_vars=2, hid_dim=8, receiver_chunk_size=2, **kwargs)


@pytest.mark.parametrize('backend', ['dense', 'sparse_csr'])
@pytest.mark.parametrize('kind,depth', [('gcn',1), ('cheb',1), ('cheb',2), ('cheb',3), ('cheb',4)])
def test_layer_matches_original_strgcn_operator(backend, kind, depth):
    torch.manual_seed(7)
    layer = DirectSTRGCNLayer(8, 4, 2, 0., True, True, 'rbf', kind, depth, backend, 2)
    b = graph()
    ids, tt = b['batch_var_idx'], b['batch_timestamp']
    valid, query = b['batch_pad_mask'].bool(), b['batch_pred_mask'].bool()
    x = torch.randn(1,7,8)
    p = time_features(layer.time_embedding, tt, valid)
    logits = layer.middle_time_temp*layer.middle_time_scale*torch.bmm(p,p.transpose(1,2))
    u, v = layer.learnable_adj_weight_U(ids), layer.learnable_adj_weight_V(ids)
    logits = logits + layer.middle_val_temp*layer.middle_val_scale*torch.bmm(u,v.transpose(1,2))
    legal = valid[:,:,None] & (valid & ~query)[:,None,:] & (tt[:,None,:] <= tt[:,:,None])
    adjacency = masked_softmax(logits, legal)
    wt = layer.middle_weight_T(p)
    if kind == 'cheb':
        ws = torch.stack([embedding(ids[0]) for embedding in layer.middle_weight_S_Embedding])
        raw = low_rank_cheb_graph_conv(x,adjacency,wt,ws,depth,layer.middle_bias)
    else:
        ws = layer.middle_weight_S_Embedding[0](ids)
        raw = low_rank_graph_conv(x,adjacency,wt,ws,layer.middle_bias)
    expected = layer.norm(layer.lambda_*raw.relu()+(1-layer.lambda_)*x)
    actual = layer(x,ids,tt,valid,query,joint_sparse_plan(tt,valid,query))
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('embedding', ['rbf','rope','transformer'])
@pytest.mark.parametrize('kind,depth', [('gcn',1),('cheb',2),('cheb',3)])
def test_csr_dense_predictions_and_all_gradients(embedding,kind,depth):
    args = dict(time_embedding_type=embedding,gcn_type=kind,gcn_depth=depth)
    dense = direct(relation_backend='dense', **args)
    sparse = direct(relation_backend='sparse_csr', **args)
    sparse.load_state_dict(dense.state_dict())
    b = {k:v.repeat(2,1) for k,v in graph().items()}
    b['batch_pad_mask'][1,0] = 0
    b['batch_value'][:,4:6] = torch.nan
    a,c = dense(**b),sparse(**b)
    torch.testing.assert_close(a,c,rtol=3e-5,atol=3e-6)
    a[:,4:6].square().sum().backward()
    c[:,4:6].square().sum().backward()
    for (name,p),(other,q) in zip(dense.named_parameters(),sparse.named_parameters()):
        assert name == other
        if p.grad is None:
            assert q.grad is None or (q.grad==0).all(),name
        else:
            assert q.grad is not None,name
            torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=5e-5,msg=name)


def test_joint_plan_includes_cross_variable_and_both_role_directions():
    b = graph()
    order = torch.tensor([6,3,5,0,2,4,1])
    b = {k:v[:,order] for k,v in b.items()}
    tt = b['batch_timestamp']
    valid,query = b['batch_pad_mask'].bool(),b['batch_pred_mask'].bool()
    expected = valid[:,:,None] & (valid & ~query)[:,None,:] & (tt[:,None,:]<=tt[:,:,None])
    actual = torch.zeros_like(expected)
    for group in joint_sparse_plan(tt,valid,query):
        for node,degree in zip(group.receivers,group.degrees):
            actual[group.batch,node,group.sources[:int(degree)]] = True
    assert torch.equal(actual,expected)
    # Historical Y (original node 1) can send to X (original node 2).
    assert actual[0,(order==2).nonzero().item(),(order==1).nonzero().item()]
    # Historical X can send to future Y, but query Y never sends back.
    assert actual[0,(order==4).nonzero().item(),(order==0).nonzero().item()]
    assert not actual[0,:,(order==4).nonzero().item()].any()


@pytest.mark.parametrize('backend', ['dense','sparse_csr'])
@pytest.mark.parametrize('depth', [2,3])
def test_no_roles_or_query_values_and_later_observations_cannot_leak(backend,depth):
    m = direct(relation_backend=backend,gcn_depth=depth).eval()
    b = graph()
    expected = m(**b)
    b['batch_node_type'] = torch.full_like(b['batch_node_type'],-999)
    torch.testing.assert_close(m(**b),expected)
    del b['batch_node_type']
    b['batch_value'][:,4:6] = torch.nan
    torch.testing.assert_close(m(**b),expected)
    b['batch_value'].requires_grad_()
    m(**b)[:,4:6].square().sum().backward()
    assert (b['batch_value'].grad[:,4:6]==0).all()
    assert b['batch_value'].grad[:,6].abs().sum()>0  # known future input influences final query
    assert torch.isfinite(m.encoder.weight.grad).all()
    with torch.no_grad():
        b['batch_value'][0,6] += 100
        changed = m(**b)
    torch.testing.assert_close(changed[:,:5],expected[:,:5])
    assert not torch.allclose(changed[:,5],expected[:,5])
    # Role-free propagation also lets another historical Y variable affect Y.
    b = graph(); b['batch_var_idx'][0,1] = 0
    b['batch_value'].requires_grad_()
    m(**b)[:,4:6].sum().backward()
    assert b['batch_value'].grad[0,1].abs()>0


@pytest.mark.parametrize('backend', ['dense','sparse_csr'])
def test_padding_query_only_and_permutation(backend):
    m,b = direct(relation_backend=backend).eval(),graph()
    expected = m(**b)
    perm = torch.tensor([6,3,5,0,2,4,1])
    torch.testing.assert_close(m(**{k:v[:,perm] for k,v in b.items()}),expected[:,perm])
    for key,value in b.items():
        fill = -999 if key in ('batch_var_idx','batch_node_type') else float('nan')
        if key in ('batch_pad_mask','batch_pred_mask'):
            fill = 0
        b[key] = torch.cat([value,torch.full((1,2),fill,dtype=value.dtype)],dim=1)
    result = m(**b)
    torch.testing.assert_close(result[:,:7],expected)
    assert (result[:,7:]==0).all()
    b = {k:v[:,4:5] for k,v in graph().items()}
    result = m(**b)
    assert torch.isfinite(result).all()
    result.sum().backward()
    b['batch_pad_mask'].zero_()
    assert (m(**b)==0).all()


@pytest.mark.parametrize('no_exogenous', [False,True])
def test_full_training_cli_dag_space_and_known_future(tmp_path,no_exogenous):
    path,_,_ = archive(tmp_path)
    c = yaml.safe_load((ROOT/'config/STRGCNDirect/processed/BE.yaml').read_text())
    c['global_config']['exen_data_path'] = str(path)
    c['data_config']['data_processor_params'].update(history_len=7,forecast_len=3,target_names=['Y'])
    c['train_config']['trainer_params']['max_epoch_num'] = 1
    config = tmp_path/'config.yaml';config.write_text(yaml.safe_dump(c))
    command = [sys.executable,'main.py','--config_path',str(config),'--device','cpu',
               '--model_save_dir_path',str(tmp_path/'model'),'--result_save_dir_path',str(tmp_path/'result')]
    if no_exogenous:
        command.append('--no-exogenous')
    run = subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=60)
    assert run.returncode==0,run.stdout+run.stderr
    report = json.loads((tmp_path/'result/metrics.json').read_text())
    assert report['protocol']['model_name']=='STRGCNDirect'
    assert report['protocol']['relation_scope']=='all_known_variables_to_all_receivers_causal'
    assert report['protocol']['known_future_exogenous_var_ids']==([] if no_exogenous else [1,2])
    mask = np.load(tmp_path/'result/test_y_mask.npy').astype(bool)
    ids = np.load(tmp_path/'result/test_var_idx.npy')
    assert mask.any() and (ids[mask]==0).all()
    from utils.dag_evaluation import DAGEvaluationScaler
    scaler = DAGEvaluationScaler().load(tmp_path/'model/evaluation_scaler.npz')
    pred = np.load(tmp_path/'result/test_y_pred.npy')
    true = np.load(tmp_path/'result/test_y_true.npy')
    windows = np.load(tmp_path/'result/test_window_id.npy')
    errors = []
    for wid in np.unique(windows):
        chosen = (windows==wid)&mask
        errors.append(np.mean(((pred[chosen]-true[chosen])/scaler.scale_[ids[chosen]])**2))
    np.testing.assert_allclose(report['test']['mse_norm'],np.mean(errors))


def test_all_processed_configs_are_role_free_and_match_exen_tasks():
    configs = list((ROOT/'config/STRGCNDirect/processed').glob('*.yaml'))
    assert len(configs)==13
    for path in configs:
        c = yaml.safe_load(path.read_text())
        previous = yaml.safe_load((ROOT/'config/ExEn/processed'/path.name).read_text())
        assert c['data_config']==previous['data_config']
        assert c['model_config']['model_name']=='STRGCNDirect'
        params = c['model_config']['model_params']
        assert params['gcn_type']=='cheb' and params['gcn_depth']==2 and params['n_layer']==3
        assert params['relation_backend']=='sparse_csr'


@pytest.mark.parametrize('include_dist', ['0','1'])
def test_batch_command_routes_to_new_configs_and_results(tmp_path,include_dist):
    import os
    fake = tmp_path/'python'
    fake.write_text(f'#!{sys.executable}\nimport json,os,sys\nwith open(os.environ["COMMAND_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    log = tmp_path/'commands.jsonl'
    env = dict(os.environ,PYTHON=str(fake),DEVICE='cuda:0',COMMAND_LOG=str(log),INCLUDE_DIST=include_dist)
    run = subprocess.run(['bash','scripts/run_strgcn_direct.sh'],cwd=ROOT,env=env,
                         text=True,capture_output=True,timeout=30)
    assert run.returncode==0,run.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands)==12+int(include_dist)
    names = []
    for command in commands:
        path = command[command.index('--config_path')+1]
        name = Path(path).stem;names.append(name)
        assert path==f'config/STRGCNDirect/processed/{name}.yaml'
        assert command[-2:]==['--device','cuda:0']
        assert f'results/strgcn_direct_known_future/{name}' in command
        assert f'model_states/strgcn_direct_known_future/{name}' in command
    assert ('DIST' in names)==(include_dist=='1')
