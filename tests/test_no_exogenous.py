import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
import yaml
from datasets import ExEnProcessedDataset
from models import ExEn
from utils.scaler import ExEnProcessedMinMaxScaler
from test_exen_processed import archive


@pytest.mark.parametrize('split',['train','valid','test'])
def test_no_x_removes_both_hist_and_future_and_preserves_all_queries(tmp_path,split):
    _,g,data=archive(tmp_path)
    scaler=ExEnProcessedMinMaxScaler();scaler.fit(data['train_data'])
    packet=scaler.transform(data[f'{split}_data'])
    full=ExEnProcessedDataset(packet,type=split,global_args=g)
    only_y=ExEnProcessedDataset(packet,type=split,global_args=g,use_exogenous=False)
    assert full.packet['windows']==only_y.packet['windows']
    assert len(full)==len(only_y)
    assert not only_y.known_future_exogenous_var_ids
    for i in range(len(full)):
        a,b=full[i],only_y[i]
        expected=a[a[:,2]==0]  # target Y raw variable ID
        torch.testing.assert_close(expected,b)
        assert (b[:,2]==0).all()
        assert torch.equal(a[a[:,3].bool()&(a[:,2]==0)],b[b[:,3].bool()])


@pytest.mark.parametrize('backend',['dense','sparse_csr'])
def test_no_x_finite_training_and_independence_from_exogenous_values(tmp_path,backend):
    _,g,data=archive(tmp_path)
    scaler=ExEnProcessedMinMaxScaler();scaler.fit(data['train_data'])
    p=scaler.transform(data['train_data'])
    ds=ExEnProcessedDataset(p,type='train',global_args=g,use_exogenous=False)
    b=ds.collate_fn([ds[0],ds[1]])
    assert set(b['batch_node_type'][b['batch_pad_mask'].bool()].tolist())=={1,3}
    before=ds[0].clone()
    # Changing all X values cannot affect a Y-only sample, even in its history.
    p['values'][:,1:]+=1e5
    torch.testing.assert_close(before,ds[0])
    model=ExEn(global_args=g,hid_dim=8,relation_backend=backend,receiver_chunk_size=3)
    q=b['batch_pred_mask'].bool();b['batch_value'].requires_grad_()
    out=model(**b).squeeze(-1)
    loss=(out[q]-b['batch_value'].detach()[q]).square().mean();loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(b['batch_value'].grad).all()
    assert (b['batch_value'].grad[q]==0).all()


def test_cli_no_exogenous_records_real_mode_and_only_y(tmp_path):
    p,_,_=archive(tmp_path)
    root=Path(__file__).resolve().parents[1]
    c=yaml.safe_load((root/'config/ExEn/processed/BE.yaml').read_text())
    c['global_config']['exen_data_path']=str(p)
    c['data_config']['data_processor_params'].update(history_len=7,forecast_len=3,target_names=['Y'])
    c['train_config']['trainer_params']['max_epoch_num']=1
    cp=tmp_path/'config.yaml';cp.write_text(yaml.safe_dump(c))
    run=subprocess.run([sys.executable,'main.py','--config_path',str(cp),'--no-exogenous','--device','cpu',
        '--model_save_dir_path',str(tmp_path/'model'),'--result_save_dir_path',str(tmp_path/'result')],
        cwd=root,text=True,capture_output=True,timeout=60)
    assert run.returncode==0,run.stdout+run.stderr
    protocol=json.loads((tmp_path/'result/evaluation_protocol.json').read_text())
    assert protocol['use_exogenous'] is False
    assert protocol['forecast_inputs']=='Y_history_only'
    assert protocol['known_future_exogenous_var_ids']==[]
    valid=np.load(tmp_path/'result/test_y_mask.npy').astype(bool)
    ids=np.load(tmp_path/'result/test_var_idx.npy')
    assert valid.any() and (ids[valid]==0).all()


def test_no_exogenous_batch_script_excludes_dist(tmp_path):
    root=Path(__file__).resolve().parents[1]
    fake=tmp_path/'python'
    fake.write_text(f'#!{sys.executable}\nimport json,os,sys\nwith open(os.environ["COMMAND_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755);log=tmp_path/'commands'
    env=dict(os.environ,PYTHON=str(fake),COMMAND_LOG=str(log),DEVICE='cuda:0')
    env.pop('RUN_TAG',None)
    run=subprocess.run(['bash','scripts/run_exen_no_exogenous_except_dist.sh'],cwd=root,env=env,
                       text=True,capture_output=True,timeout=30)
    assert run.returncode==0,run.stderr
    commands=[json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands)==12
    for args in commands:
        assert '--no-exogenous' in args
        assert args[args.index('--device')+1]=='cuda:0'
        assert '/exen_no_exogenous/' in args[args.index('--result_save_dir_path')+1]
        assert Path(args[args.index('--config_path')+1]).stem!='DIST'
