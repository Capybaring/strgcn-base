"""Processed DIST windows: shared scaling/metrics, patch fidelity and isolation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml
from baseline.patch_fusion.adapter import TPatchGNNFusion, HiPatchFusion
from datasets import TPatchFusionDataset
from utils.scaler import ExEnProcessedMinMaxScaler
from test_exen_processed import archive

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['ExEnGraphTransformer', 'TPatchGNNFusion', 'HiPatchFusion']


@pytest.mark.parametrize('cls', [TPatchGNNFusion, HiPatchFusion])
@pytest.mark.parametrize('mode', ['FullExo', 'HistExo', 'NoExo'])
def test_processed_patches_keep_history_and_isolate_future(tmp_path, cls, mode):
    _, g, data = archive(tmp_path)
    scale = ExEnProcessedMinMaxScaler(); scale.fit(data['train_data'])
    ds = TPatchFusionDataset(scale.transform(data['train_data']), type='train',
        global_args=g, patch_size=2, use_exogenous=mode != 'NoExo',
        known_future_exogenous_var_ids=[] if mode == 'HistExo' else 'all')
    samples = [ds[0], ds[1]]
    b = ds.collate_fn(samples)
    assert ds.npatch == 4  # Final partial patch still contains real history.
    for i in range(2):
        historical = b['batch_pad_mask'][i].bool() & (b['batch_node_type'][i] < 2)
        assert int(b['batch_backbone_mask'][i].sum()) == int(historical.sum())
        for original, local in ds.local_id.items():
            expected = b['batch_value'][i, historical & (b['batch_var_idx'][i] == original)].sort().values
            mask = b['batch_backbone_mask'][i, ..., local].bool()
            actual = b['batch_backbone_values'][i, ..., local][mask].sort().values
            torch.testing.assert_close(expected, actual)
    if mode != 'FullExo':
        assert not b['batch_backbone_future_x'].any()
    if mode == 'NoExo':
        assert not b['batch_backbone_mask'][..., 1:].any()
    m = cls(global_args=g, hid_dim=8).eval()
    before = m(**b)
    loss = before.squeeze(-1)[b['batch_pred_mask'].bool()].square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters())
    changed = copy.deepcopy(samples)
    for sample, _ in changed:
        future_y = sample[:, 3].bool() & (sample[:, 2] == 0)
        sample[future_y, 1] = 1e8
    modified = ds.collate_fn(changed)
    torch.testing.assert_close(before, m(**modified))
    if mode == 'FullExo':
        # A supplied X later than the first query must not alter that query.
        sample, origin = samples[0]
        late = (sample[:, 3].bool()) & (sample[:, 2] == 1) & (sample[:, 0] > .8)
        assert late.any()
        changed = sample.clone(); changed[late, 1] += 1e5
        one = ds.collate_fn([samples[0]])
        other = ds.collate_fn([(changed, origin)])
        early = one['batch_pred_mask'].bool() & (one['batch_timestamp'] <= .8)
        assert early.any()
        torch.testing.assert_close(m(**one).squeeze(-1)[early], m(**other).squeeze(-1)[early])


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('mode', ['FullExo', 'HistExo', 'NoExo'])
def test_dist_cli_metrics_and_chronological_exports(tmp_path, model, mode):
    path, _, data = archive(tmp_path)
    c = yaml.safe_load((ROOT / f'config/{model}/processed/DIST.yaml').read_text())
    c['global_config']['exen_data_path'] = str(path)
    c['data_config']['data_processor_params'].update(history_len=7, forecast_len=3,
        target_names=['Y', 'X1'], max_windows_per_split=2)
    if model == 'ExEnGraphTransformer':
        c['model_config']['model_params'].update(hid_dim=8, n_layer=1, n_cross_layer=1, num_heads=1)
    else:
        c['data_config']['dataset_params']['patch_size'] = 2
        c['model_config']['model_params']['hid_dim'] = 8
    c['train_config']['trainer_params']['max_epoch_num'] = 1
    config = tmp_path / 'run.yaml'; config.write_text(yaml.safe_dump(c))
    result, saved = tmp_path / 'result', tmp_path / 'model'
    args = [sys.executable, 'main.py', '--config_path', str(config), '--device', 'cpu',
        '--model_save_dir_path', str(saved), '--result_save_dir_path', str(result)]
    if model == 'ExEnGraphTransformer':
        args += ['--time-layers', '0', '--variable-layers', '2', '--within-variables', 'endogenous']
    if mode == 'HistExo': args.append('--history-only-exogenous')
    if mode == 'NoExo': args.append('--no-exogenous')
    run = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=90,
        env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((result / 'metrics.json').read_text())
    protocol = report['protocol']
    assert 'mse_norm' not in report['test']
    assert protocol['normalization'] == 'MinMax_on_training_windows_only_no_validation_or_test'
    assert protocol['primary_space'] == report['test']['primary_space'] == 'train_minmax_input_space'
    assert protocol['aggregation'] == 'global_mean_per_variable_then_equal_mean_available_variables'
    assert protocol['use_exogenous'] == (mode != 'NoExo')
    assert bool(protocol['known_future_exogenous_var_ids']) == (mode == 'FullExo')
    if model == 'ExEnGraphTransformer':
        assert protocol['model_parameters']['n_layer'] == 0
        assert protocol['model_parameters']['n_variable_layer'] == 2
        assert not protocol['model_parameters']['within_exogenous']
    assert protocol['history_seconds'] == 7 and protocol['forecast_seconds'] == 3
    assert (protocol['fusion'] is not None) == (model != 'ExEnGraphTransformer')
    assert not (saved / 'evaluation_scaler.npz').exists()
    assert not (result / 'test_record_id.npy').exists()
    np.testing.assert_array_equal(np.load(result / 'test_window_start_seconds.npy'),
        [w[3] for w in data['test_data']['windows']])
    pred, truth, mask, ids = [np.load(result / name) for name in
        ['test_y_pred.npy', 'test_y_true.npy', 'test_y_mask.npy', 'test_var_idx.npy']]
    errors = [pred[mask & (ids == v)].astype('float64') - truth[mask & (ids == v)].astype('float64')
              for v in [0, 1]]
    expected_mse = np.mean([np.mean(e.astype('float64') ** 2) for e in errors])
    expected_mae = np.mean([np.mean(np.abs(e.astype('float64'))) for e in errors])
    assert report['test']['mse'] == pytest.approx(expected_mse, rel=1e-6)
    assert report['test']['mae'] == pytest.approx(expected_mae, rel=1e-6)
    assert report['test']['rmse'] == pytest.approx(expected_mse ** .5, rel=1e-6)
    # The limit selects only the first two training windows; held-out maxima are larger.
    with np.load(saved / 'scaler.npz') as scale:
        assert scale['minimum'][0] == 2 and scale['ranges'][0] == 18


def test_dist_runners_select_processed_configs_and_separate_modes(tmp_path):
    fake, log = tmp_path / 'python', tmp_path / 'calls'
    fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CALLS"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    for mode in ['FullExo', 'HistExo', 'NoExo']:
        run = subprocess.run(['bash', 'scripts/run_dist.sh'], cwd=ROOT,
            env=dict(os.environ, PYTHON=str(fake), CALLS=str(log), MODE=mode, MODEL='all'),
            capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
    run = subprocess.run(['bash', 'scripts/run_patch_fusion.sh', 'dist'], cwd=ROOT,
        env=dict(os.environ, PYTHON=str(fake), CALLS=str(log), BACKBONE='all', MODE='FullExo'),
        capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    calls = [json.loads(s) for s in log.read_text().splitlines()]
    assert len(calls) == 11
    assert all('/processed/DIST.yaml' in a[a.index('--config_path') + 1] for a in calls)
    outputs = [a[a.index('--result_save_dir_path') + 1] for a in calls[:9]]
    assert len(set(outputs)) == 9
    for i, a in enumerate(calls[:9]):
        assert ('--history-only-exogenous' in a) == (3 <= i < 6)
        assert ('--no-exogenous' in a) == (6 <= i < 9)
