"""Record splits, exact task boundaries, zero semantics and oracle-input isolation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
import torch
import yaml

from data_processors import TPatchRecordProcessor
from datasets import TPatchRecordDataset
from utils.scaler import ExEnProcessedMinMaxScaler
from utils.dag_evaluation import DAGEvaluationScaler

ROOT = Path(__file__).resolve().parents[1]


def make_data(root, dataset, n=15):
    dim = dict(mimic=96, physionet=41, activity=12, ushcn=5)[dataset]
    times = dict(mimic=[0, 23, 24, 30, 48], physionet=[0, 23, 24, 30, 48],
                 activity=[0, 1000, 2999, 3000, 3999, 4000, 4999, 5000],
                 ushcn=list(np.arange(0, 48.5, .5)))[dataset]
    records = []
    for i in range(n):
        t = torch.tensor(times, dtype=torch.float32)
        x = torch.arange(len(t) * dim, dtype=torch.float32).reshape(len(t), dim) + i
        mask = torch.ones_like(x)
        if dataset == 'physionet':
            mask[:, :4] = 0
            mask[0, :4] = 1
            x[0, :4] = torch.tensor([65., 0., 170., 3.])
            x[1:, :4] = 0  # future placeholders MUST NOT be used as static values
            x[1, 4] = -1
            if i == 0:
                x[0, 2] = -1  # absent baseline cannot be filled from a future value
                x[1, 2], mask[1, 2] = 199., 1.
        # Observed zero is valid; missing nonzero is not.
        x[0, -1], mask[1, -1] = 0., 0.
        records.append((f'r{i}', t, x, mask))
    p = root / dataset / 'processed'
    p.mkdir(parents=True)
    if dataset == 'physionet':
        for i, split in enumerate('abc'):
            torch.save(records[i::3], p / f'set-{split}_0.0.pt')
        order = records[0::3] + records[1::3] + records[2::3]
    else:
        torch.save(records, p / dict(mimic='mimic.pt', activity='data.pt', ushcn='ushcn.pt')[dataset])
        order = records
    return order


def process(root, dataset, **kwargs):
    g = {'device': 'cpu'}
    data = TPatchRecordProcessor(g, dataset, data_root=root, **kwargs).process()
    return g, data


@pytest.mark.parametrize('dataset', ['mimic', 'physionet', 'activity', 'ushcn'])
def test_source_split_matches_upstream_before_chunking(tmp_path, dataset):
    order = make_data(tmp_path, dataset)
    _, data = process(tmp_path, dataset)
    seen, test = train_test_split(order, train_size=.8, random_state=42, shuffle=True)
    train, valid = train_test_split(seen, train_size=.75, random_state=42, shuffle=False)
    for split, expected in zip(('train', 'valid', 'test'), (train, valid, test)):
        assert [r[0] for r in data[f'{split}_data']['records']] == [r[0] for r in expected]
    ids = [{r[0] for r in data[f'{s}_data']['records']} for s in ('train', 'valid', 'test')]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    p = data['train_data']
    if dataset in ('mimic', 'physionet'):
        assert len(p['windows']) == len(p['records'])  # one task per patient, no global sliding
        _, a, h, b, start = p['windows'][0]
        assert (a, h, b, start) == (0, 2, 5, 0.)  # history <24; horizon includes 48
    elif dataset == 'activity':
        assert [w[4] for w in p['windows'] if w[0] == 0] == [0., 1000.]
        assert p['windows'][0][3] == 5  # first chunk excludes its 4000 ms boundary
        assert p['windows'][1][3] == 8  # final chunk includes the 5000 ms boundary
    else:
        starts = [w[4] for w in p['windows'] if w[0] == 0]
        assert starts == list(np.arange(24, dtype=float))
        assert p['windows'][-1][4] == 23.
        assert p['windows'][-1][3] == 97  # last task includes month 48


@pytest.mark.parametrize('dataset', ['mimic', 'physionet', 'activity', 'ushcn'])
def test_scaling_roles_masks_and_future_input_isolation(tmp_path, dataset):
    make_data(tmp_path, dataset)
    g, data = process(tmp_path, dataset, max_samples_per_split=2)
    packet = data['train_data']
    scale = ExEnProcessedMinMaxScaler(); scale.fit(packet)
    evaluation = DAGEvaluationScaler().fit(packet)
    for j in packet['target_ids']:
        expected = np.concatenate([x[:, j][m[:, j]] for _, _, x, m in packet['records']])
        np.testing.assert_allclose(evaluation.mean_[j], expected.mean(), rtol=1e-6)
    # Validation changes do not participate in either fitted space.
    before = scale.minimum.copy()
    data['test_data']['records'][0][2][:] = 1e8
    np.testing.assert_array_equal(scale.minimum, before)
    ds = TPatchRecordDataset(scale.transform(packet), type='train', global_args=g)
    batch = ds.collate_fn([ds[0], ds[1]])
    observed = batch['batch_pad_mask'].bool()
    query = batch['batch_pred_mask'].bool()
    assert query.any()
    assert set(batch['batch_var_idx'][query].tolist()) <= set(packet['target_ids'])
    assert ((batch['batch_node_type'] == 2) & observed).any()
    raw = TPatchRecordDataset(packet, type='train', global_args={})[0]
    assert ((raw[:, 1] == 0) & (raw[:, 2] == len(packet['names']) - 1)).any()
    if dataset == 'physionet':
        for r, t, x, mask in packet['records']:
            assert not mask[1, 4] and mask[0, 1]  # -1 invalid, female=0 valid
        static_future = observed & (batch['batch_node_type'] == 2) & (batch['batch_var_idx'] < 4)
        values = scale.inverse_transform_mx(batch['batch_value'].numpy(), batch['batch_var_idx'].numpy())
        np.testing.assert_allclose(values[static_future.numpy() & (batch['batch_var_idx'].numpy() == 1)], 0.)
    for known, use_exogenous in [([], True), ('all', False)]:
        ablation = TPatchRecordDataset(scale.transform(packet), type='train', global_args={},
                    known_future_exogenous_var_ids=known, use_exogenous=use_exogenous)
        b = ablation.collate_fn([ablation[0]])
        assert not ((b['batch_node_type'] == 2) & b['batch_pad_mask'].bool()).any()
        if not use_exogenous:
            assert set(b['batch_var_idx'][b['batch_pad_mask'].bool()].tolist()) <= set(packet['target_ids'])


@pytest.mark.parametrize('model_name', ['ExEn', 'STRGCNDirect', 'ExEnAligned', 'ExEnGraphTransformer'])
def test_graph_backbones_never_observe_future_targets(tmp_path, model_name):
    import models
    make_data(tmp_path, 'activity')
    g, data = process(tmp_path, 'activity', max_samples_per_split=1)
    ds = TPatchRecordDataset(data, type='train', global_args=g)
    b = ds.collate_fn([ds[0]])
    params = dict(hid_dim=8, node_dim=4, n_layer=1, dropout=0., receiver_chunk_size=32)
    if model_name == 'ExEnGraphTransformer':
        params.update(n_cross_layer=1, num_heads=2)
    elif model_name == 'ExEnAligned':
        params.update(n_cross_layer=1)
    model = getattr(models, model_name)(global_args=g, **params).eval()
    query = b['batch_pred_mask'].bool()
    inputs = {**b, 'batch_value': torch.where(query, 0, b['batch_value'])}
    out = model(**inputs).squeeze(-1)
    altered = {**b, 'batch_value': b['batch_value'].clone()}
    altered['batch_value'][query] += 100000.
    altered['batch_value'] = torch.where(query, 0, altered['batch_value'])
    torch.testing.assert_close(out, model(**altered).squeeze(-1))
    loss = (out[query] - b['batch_value'][query]).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('dataset', ['mimic', 'physionet', 'activity', 'ushcn'])
def test_cli_one_epoch_exports_record_protocol(tmp_path, dataset):
    make_data(tmp_path, dataset)
    c = yaml.safe_load((ROOT / f'config/ExEnGraphTransformer/tpatch/{dataset}.yaml').read_text())
    c['data_config']['data_processor_params']['max_samples_per_split'] = 1
    c['model_config']['model_params'].update(hid_dim=8, node_dim=4, n_layer=1, n_cross_layer=1, num_heads=2)
    c['train_config']['trainer_params']['max_epoch_num'] = 1
    config = tmp_path / 'config.yaml'; config.write_text(yaml.safe_dump(c))
    command = [sys.executable, 'main.py', '--config_path', str(config), '--device', 'cpu',
        '--tpatch-data-root', str(tmp_path), '--model_save_dir_path', str(tmp_path / 'model'),
        '--result_save_dir_path', str(tmp_path / 'result')]
    # Exercise CLI Y-only path too, with one of the four end-to-end runs.
    if dataset == 'activity':
        command.append('--no-exogenous')
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((tmp_path / 'result/evaluation_protocol.json').read_text())
    assert report['tpatch']['split_seed'] == 42
    assert report['native_time_unit'] == dict(mimic='hours', physionet='hours', activity='milliseconds', ushcn='months')[dataset]
    assert 'observed zeros retained' in report['missingness']
    assert (tmp_path / 'model/record_splits.npz').exists()
    assert (tmp_path / 'result/test_record_id.npy').exists()
    assert not (tmp_path / 'result/test_window_start_seconds.npy').exists()
    metrics = json.loads((tmp_path / 'result/metrics.json').read_text())
    assert np.isfinite(metrics['test']['mse_norm'])


def test_activity_reversals_and_duplicate_timestamps_preserve_observations(tmp_path):
    make_data(tmp_path, 'activity')
    path = tmp_path / 'activity/processed/data.pt'
    records = torch.load(path, weights_only=True)
    for i, (rid, t, x, m) in enumerate(records):
        # Reverse two rows and repeat one time: input record order is not reliable.
        order = torch.tensor([0, 2, 1, 3, 4, 5, 6, 7])
        t = t[order]; x = x[order]; m = m[order]
        t[4] = t[3]
        records[i] = (rid, t, x, m)
    torch.save(records, path)
    _, data = process(tmp_path, 'activity')
    for split in ('train', 'valid', 'test'):
        for _, t, x, mask in data[f'{split}_data']['records']:
            assert np.all(np.diff(t) >= 0)
            assert np.sum(t == 3000) == 2  # retained, never silently deduplicated
            assert len(t) == 8


def test_invalid_baseline_never_uses_future_measurement(tmp_path):
    make_data(tmp_path, 'physionet')
    g, data = process(tmp_path, 'physionet')
    packet = next(data[f'{s}_data'] for s in ('train', 'valid', 'test')
                  if any(r[0] == 'r0' for r in data[f'{s}_data']['records']))
    ds = TPatchRecordDataset(packet, type='train', global_args=g)
    index = next(i for i, w in enumerate(packet['windows']) if packet['records'][w[0]][0] == 'r0')
    sample = ds[index]
    assert not (sample[:, 2] == 2).any()  # no height=199 backfill from t=23


def test_record_scaler_state_roundtrip_and_unseen_features(tmp_path):
    make_data(tmp_path, 'activity')
    _, data = process(tmp_path, 'activity')
    packet = data['train_data']
    for _, _, _, mask in packet['records']:
        mask[:, 0] = False
    scale = ExEnProcessedMinMaxScaler(); scale.fit(packet)
    assert scale.unseen_variable_ids == [0]
    scale.save(tmp_path / 'input.npz')
    restored = ExEnProcessedMinMaxScaler(); restored.load(tmp_path / 'input.npz')
    assert restored.unseen_variable_ids == [0]
    evaluation = DAGEvaluationScaler().fit(packet)
    assert evaluation.count_[0] == 0 and evaluation.scale_[0] == 1
    evaluation.save(tmp_path / 'eval.npz')
    report = DAGEvaluationScaler().load(tmp_path / 'eval.npz').manifest()
    assert report['fit_scope'] == 'training_records_only'
    assert 'observed zero retained' in report['zero_rule']


def test_runner_routes_four_configs_and_ablation(tmp_path):
    fake = tmp_path / 'python'
    log = tmp_path / 'calls'
    fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON=str(fake), CALLS=str(log), DEVICE='cpu', NO_EXOGENOUS='1', TPATCH_DATA_ROOT='/tmp/reference-data')
    run = subprocess.run(['bash', 'scripts/run_tpatch_oracle.sh'], cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 4
    for dataset, args in zip(('mimic', 'physionet', 'activity', 'ushcn'), calls):
        assert f'config/ExEnGraphTransformer/tpatch/{dataset}.yaml' in args
        assert '--no-exogenous' in args
        assert args[args.index('--tpatch-data-root') + 1] == '/tmp/reference-data'


def test_source_files_are_not_rewritten(tmp_path):
    import hashlib
    make_data(tmp_path, 'physionet')
    paths = sorted(tmp_path.glob('physionet/processed/*.pt'))
    before = [hashlib.sha256(p.read_bytes()).digest() for p in paths]
    process(tmp_path, 'physionet')
    assert before == [hashlib.sha256(p.read_bytes()).digest() for p in paths]
