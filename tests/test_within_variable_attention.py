"""Within-role cross-variable edges, independent ablations and causal packing."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from models import ExEnGraphTransformer
from models.STRGCN.SparseRelations import build_within_variable_relations
from models.STRGCN.PackedAttention import pack_attention_plan

ROOT = Path(__file__).resolve().parents[1]


def observations():
    return dict(batch_timestamp=torch.tensor([[0., .1, .2, .2, .6, .85, .8, .9]]),
        batch_var_idx=torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
        batch_node_type=torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]]),
        batch_value=torch.tensor([[1., 2., 3., 4., 5., 6., 77., 88.]]),
        batch_pad_mask=torch.ones(1, 8),
        batch_pred_mask=torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.]]))


def model(**kwargs):
    return ExEnGraphTransformer(num_vars=4, exogenous_var_indices=[0, 1], hid_dim=8,
        num_heads=2, n_cross_layer=1, dropout=0., n_variable_layer=1, **kwargs)


def test_edges_match_independent_dense_definition():
    b = observations()
    t, v, role = [b[k][0] for k in ['batch_timestamp', 'batch_var_idx', 'batch_node_type']]
    groups = build_within_variable_relations(t[None], v[None], role[None], torch.ones(1, 8).bool())
    for relation in [3, 4]:
        actual = torch.zeros(8, 8).bool()
        for group in groups:
            if group.relation == relation:
                for receiver, degree in zip(group.receivers, group.degrees):
                    actual[receiver, group.sources[:degree]] = True
        expected = torch.zeros_like(actual)
        for receiver in range(8):
            for source in range(8):
                same_role = ((role[receiver] in [0, 2]) == (role[source] in [0, 2]))
                selected = (role[receiver] in [0, 2]) if relation == 3 else (role[receiver] in [1, 3])
                expected[receiver, source] = (selected and same_role and role[source] != 3
                    and v[source] != v[receiver] and t[source] <= t[receiver])
        assert torch.equal(actual, expected)
        assert actual.any()


@pytest.mark.parametrize('within_x,within_y', [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize('checkpoint', [True, False])
def test_packed_streaming_outputs_and_gradients(within_x, within_y, checkpoint):
    torch.manual_seed(14)
    a = model(within_exogenous=within_x, within_endogenous=within_y, checkpoint_attention=checkpoint)
    c = copy.deepcopy(a)
    b = observations()
    relations = tuple(r for r, enabled in [(3, within_x), (4, within_y)] if enabled)
    plan = pack_attention_plan(b['batch_timestamp'], b['batch_var_idx'], b['batch_node_type'],
        b['batch_pad_mask'].bool(), within_relations=relations, receiver_chunk=2, max_score_elements=16)
    before = a(**b); after = c(**b, batch_attention_plan=plan)
    torch.testing.assert_close(before, after, atol=3e-6, rtol=3e-5)
    before.square().sum().backward(); after.square().sum().backward()
    for (name, p), (_, q) in zip(a.named_parameters(), c.named_parameters()):
        if p.grad is None:
            assert q.grad is None or not q.grad.any(), name
        else:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=5e-5, rtol=5e-4, msg=name)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.variable_layers.parameters())


@pytest.mark.parametrize('packed', [False, True])
def test_queries_and_later_x_cannot_leak(packed):
    b = observations(); m = model().eval()
    extra = {}
    if packed:
        # The model must also accept legacy plans built without within-group edges.
        extra['batch_attention_plan'] = pack_attention_plan(b['batch_timestamp'], b['batch_var_idx'],
            b['batch_node_type'], b['batch_pad_mask'].bool())
    before = m(**b, **extra)
    b['batch_value'][:, 6:] = torch.nan
    torch.testing.assert_close(before, m(**b, **extra))
    b['batch_value'][0, 5] += 100
    changed = m(**b, **extra)
    torch.testing.assert_close(before[:, 6], changed[:, 6])


def test_zero_depths_disable_stages_and_disabled_group_is_unchanged():
    b = observations()
    m = model(n_layer=0, within_exogenous=False)
    assert len(m.temporal_layers) == 0 and len(m.variable_layers) == 1
    layer = m.variable_layers[0]
    h = torch.randn(1, 8, 8)
    plan = build_within_variable_relations(b['batch_timestamp'], b['batch_var_idx'],
        b['batch_node_type'], b['batch_pad_mask'].bool(), (4,))
    out = layer(h, None, b['batch_var_idx'], None, b['batch_pad_mask'].bool(),
        b['batch_timestamp'], b['batch_node_type'], sparse_plan=plan)
    torch.testing.assert_close(h[:, :2], out[:, :2])
    old = ExEnGraphTransformer(num_vars=4, exogenous_var_indices=[0, 1], hid_dim=8)
    restored = ExEnGraphTransformer(num_vars=4, exogenous_var_indices=[0, 1], hid_dim=8, n_variable_layer=0)
    restored.load_state_dict(old.state_dict(), strict=True)
    assert not restored.variable_layers
    assert torch.isfinite(m(**b)).all()


@pytest.mark.parametrize('script,dataset,env_model', [
    ('run_tpatch_oracle.sh', 'activity', 'ExEnGraphTransformer'),
    ('run_dist.sh', None, 'exen'),
    ('run_exen_graph_transformer.sh', None, None)])
def test_runner_forwards_ablation_flags_and_separates_outputs(tmp_path, script, dataset, env_model):
    fake, log = tmp_path/'python', tmp_path/'calls'
    fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CALLS"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON=str(fake), CALLS=str(log), TIME_LAYERS='0',
        VARIABLE_LAYERS='2', CROSS_LAYERS='1', WITHIN_VARIABLES='endogenous')
    env.pop('RUN_TAG', None)
    if env_model: env['MODEL'] = env_model
    cmd = ['bash', f'scripts/{script}'] + ([dataset] if dataset else [])
    run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    for call in calls:
        for flag, value in [('--time-layers','0'), ('--variable-layers','2'),
                            ('--cross-layers','1'), ('--within-variables','endogenous')]:
            assert call[call.index(flag)+1] == value
        assert '_t0_v2_c1_endogenous' in call[call.index('--result_save_dir_path')+1]


def test_cli_none_resolves_zero_variable_depth(tmp_path):
    import yaml
    from test_exen_processed import archive
    path, _, _ = archive(tmp_path)
    c = yaml.safe_load((ROOT/'config/ExEnGraphTransformer/processed/DIST.yaml').read_text())
    c['global_config']['exen_data_path'] = str(path)
    c['data_config']['data_processor_params'].update(history_len=7, forecast_len=3,
        target_names=['Y'], max_windows_per_split=1)
    c['train_config']['trainer_params']['max_epoch_num'] = 1
    config = tmp_path/'config.yaml'; config.write_text(yaml.safe_dump(c))
    result = tmp_path/'result'
    run = subprocess.run([sys.executable, 'main.py', '--config_path', str(config),
        '--device', 'cpu', '--time-layers', '0', '--cross-layers', '0',
        '--variable-layers', '1', '--within-variables', 'none',
        '--model_save_dir_path', str(tmp_path/'model'), '--result_save_dir_path', str(result)],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    p = json.loads((result/'evaluation_protocol.json').read_text())['model_parameters']
    assert p['n_layer'] == p['n_variable_layer'] == p['n_cross_layer'] == 0
    assert not p['within_exogenous'] and not p['within_endogenous']
