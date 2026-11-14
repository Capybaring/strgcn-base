"""Sequential known-node memory, ordered patterns and physical-time isolation."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from models import ExEnGraphTransformer
from models.STRGCN.RollingWindows import pack_rolling_attention_plan
from test_within_variable_attention import observations

ROOT = Path(__file__).resolve().parents[1]


def model(**kwargs):
    torch.manual_seed(51)
    return ExEnGraphTransformer(num_vars=4, exogenous_var_indices=[0, 1], hid_dim=8,
        num_heads=2, n_layer=1, n_variable_layer=1, n_cross_layer=1,
        rolling_window_size=.2, dropout=0., checkpoint_attention=False, **kwargs)


def plan(batch, **kwargs):
    return pack_rolling_attention_plan(batch['batch_timestamp'], batch['batch_var_idx'],
        batch['batch_node_type'], batch['batch_pad_mask'].bool(), .2,
        bounds=torch.tensor([[0., .6, 1.]]).expand(len(batch['batch_value']), -1),
        within_relations=(3, 4), receiver_chunk=2, max_score_elements=32, **kwargs)


def test_schedule_retains_empty_windows_and_physical_time_coverage():
    b = observations(); p = plan(b)
    assert len(p.windows) == 5
    assert not p.windows[2].receivers.any()  # [.4,.6) is not compressed away
    assert torch.equal(sum(w.receivers.long() for w in p.windows), b['batch_pad_mask'].long())
    for w in p.windows:
        for bucket in w.plan.buckets:
            assert (b['batch_node_type'].flatten()[bucket.source]
                    .masked_select(torch.arange(bucket.source.shape[1])[None] < bucket.degrees.max(-1).values[:, None]) != 3).all()
            assert w.receivers.flatten()[bucket.destinations].all()
    # The first interval starts before the first observation, not at that observation.
    assert p.windows[0].bounds[0, 0] == 0
    moved = p.to('cpu')
    assert len(moved.windows) == len(p.windows)


@pytest.mark.parametrize('backend', ['sdpa', 'reference'])
def test_queries_and_later_observations_never_leak_within_or_between_windows(backend):
    b = observations(); m = model(attention_backend=backend).eval(); p = plan(b)
    before = m(**b, batch_attention_plan=p)
    b['batch_value'][:, 6:] = torch.nan
    torch.testing.assert_close(before, m(**b, batch_attention_plan=p))
    b['batch_value'][:, 6:] = float('inf')
    torch.testing.assert_close(before, m(**b, batch_attention_plan=p))
    # .85 X is in the SAME window as .8 query, but cannot affect that query.
    b['batch_value'][0, 5] += 80
    after = m(**b, batch_attention_plan=p)
    torch.testing.assert_close(before[:, :5], after[:, :5])
    torch.testing.assert_close(before[:, 6], after[:, 6])
    assert not torch.allclose(before[:, 7], after[:, 7])
    b = observations(); b['batch_value'].requires_grad_()
    m(**b, batch_attention_plan=p)[:, 6:].square().sum().backward()
    assert (b['batch_value'].grad[:, 6:] == 0).all()
    assert b['batch_value'].grad[:, :4].abs().sum() > 0


def test_frozen_past_patterns_condition_every_stage_and_keep_gradient_paths():
    b = observations(); m = model(); p = plan(b)
    states, summaries, updates = [], [], []
    handle = m.window_memory.register_forward_hook(lambda *args: None)
    original = m.window_memory.predict
    def predict(tokens, masks, bounds):
        state = original(tokens, masks, bounds)
        states.append(state)
        return state
    m.window_memory.predict = predict
    original_summary = m.window_memory.summarize
    def summarize(stages, tt, ids, role, known, bounds):
        assert not (known & (role == 3)).any()
        token = original_summary(stages, tt, ids, role, known, bounds)
        token.retain_grad(); summaries.append(token)
        return token
    m.window_memory.summarize = summarize
    for layer in [*m.temporal_layers, *m.variable_layers, *m.cross_layers]:
        def hook(module, args, kwargs, output):
            h = args[0]; receiver = kwargs['receiver_mask']
            torch.testing.assert_close(output[~receiver], h[~receiver])
            assert kwargs['window_state'] is states[-1]
            updates.append(receiver.clone())
        layer.register_forward_hook(hook, with_kwargs=True)
    out = m(**b, batch_attention_plan=p)
    out[:, 6:].square().sum().backward()
    handle.remove()
    assert len(states) == len(summaries) == 5
    assert len(updates) == 15
    assert summaries[0].grad is not None and summaries[0].grad.abs().sum() > 0
    assert m.window_memory.interactions.weight.grad.abs().sum() > 0
    assert m.window_memory.attention.in_proj_weight.grad.abs().sum() > 0
    for layers in (m.temporal_layers, m.variable_layers, m.cross_layers):
        assert sum(x.grad.abs().sum() for layer in layers for x in layer.window_query.parameters()) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_window_history_order_changes_conditioning_even_with_identical_times():
    m = model(); memory = m.window_memory
    a, b = torch.randn(1, 8), torch.randn(1, 8)
    # Tokens include their boundary encoding, so exchanging patterns at fixed
    # temporal slots is observable (not an unordered bag of observations).
    first = memory._boundary(torch.tensor([[0., .2]]))
    second = memory._boundary(torch.tensor([[.2, .4]]))
    keep = [torch.ones(1, dtype=torch.bool)] * 2
    query = torch.tensor([[.4, .6]])
    x = memory.predict([a + first, b + second], keep, query)
    y = memory.predict([b + first, a + second], keep, query)
    assert not torch.allclose(x, y, atol=1e-7, rtol=1e-7)


def test_batch_padding_permutation_no_x_and_backend_gradients():
    b = {k: v.repeat(2, 1) for k, v in observations().items()}
    b['batch_pad_mask'][1, [0, 1, 4, 5]] = 0
    p = plan(b)
    a = model(attention_backend='sdpa'); c = model(attention_backend='reference')
    c.load_state_dict(a.state_dict())
    x, y = a(**b, batch_attention_plan=p), c(**b, batch_attention_plan=p)
    torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-5)
    x[:, 6:].square().sum().backward(); y[:, 6:].square().sum().backward()
    for (name, v), (_, w) in zip(a.named_parameters(), c.named_parameters()):
        if v.grad is not None:
            assert w.grad is not None, name
            torch.testing.assert_close(v.grad, w.grad, atol=5e-5, rtol=5e-4, msg=name)
    order = torch.arange(7, -1, -1)
    permuted = {k: v[:, order] for k, v in b.items()}
    torch.testing.assert_close(x.detach()[:, order], a(**permuted, batch_attention_plan=plan(permuted)))
    for key, value in b.items():
        fill = 0 if key in ('batch_pad_mask', 'batch_pred_mask') else float('nan')
        # var IDs and types are integer tensors; invalid values are sanitized.
        if not value.is_floating_point(): fill = -99
        b[key] = torch.cat((value, torch.full((2, 2), fill, dtype=value.dtype)), 1)
    z = a(**b, batch_attention_plan=plan(b))
    torch.testing.assert_close(x.detach(), z[:, :8])
    assert (z[:, 8:] == 0).all()


def test_all_padding_record_and_empty_source_windows_are_finite():
    b = {k: v.repeat(2, 1) for k, v in observations().items()}
    b['batch_pad_mask'][1] = 0
    p = plan(b); m = model()
    out = m(**b, batch_attention_plan=p)
    assert torch.isfinite(out).all() and not out[1].any()
    out.square().sum().backward()
    assert all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters())


def test_observation_summary_is_independent_of_all_query_representations():
    m = model(); b = observations(); known = b['batch_node_type'] != 3
    stages = [torch.randn(1, 8, 8, requires_grad=True) for _ in range(3)]
    bounds = torch.tensor([[0., 1.]])
    token = m.window_memory.summarize(stages, b['batch_timestamp'], b['batch_var_idx'],
        b['batch_node_type'], known, bounds)
    changed = [torch.where(known[..., None], h, h + 100) for h in stages]
    torch.testing.assert_close(token, m.window_memory.summarize(changed, b['batch_timestamp'],
        b['batch_var_idx'], b['batch_node_type'], known, bounds))
    token.square().sum().backward()
    assert all(not h.grad[~known].any() for h in stages)


def test_initial_widths_match_dataset_time_units():
    from models.STRGCN.RollingWindows import _intervals
    # Width = native history duration / 8; forecast uses the same width and
    # retains its final partial window without changing the forecast boundary.
    cases = [('tpatch/mimic', 48, 24, 24, 3, 16),
             ('tpatch/physionet', 48, 24, 24, 3, 16),
             ('tpatch/activity', 4, 3, 1, .375, 11),
             ('tpatch/ushcn', 48, 24, 1, 3, 9),
             ('processed/DIST', 9000, 7200, 1800, 900, 10)]
    for name, scale, history, forecast, width, expected in cases:
        c = yaml.safe_load((ROOT / f'config/ExEnGraphTransformer/{name}.yaml').read_text())
        normalized = c['model_config']['model_params']['rolling_window_size']
        assert normalized * scale == pytest.approx(width)
        assert width == pytest.approx(history / 8)
        windows = _intervals(0., history / scale, (history + forecast) / scale, normalized)
        assert len(windows) == expected
        assert sum(right <= history / scale for left, right in windows) == 8
        assert windows[8][0] == pytest.approx(history / scale)
        assert windows[-1][1] == pytest.approx((history + forecast) / scale)


@pytest.mark.parametrize('width', [0, -1, float('nan'), float('inf'), True])
def test_invalid_widths_are_rejected(width):
    with pytest.raises(ValueError, match='rolling_window_size'):
        ExEnGraphTransformer(num_vars=2, exogenous_var_indices=[0], rolling_window_size=width)


def test_final_endpoint_and_forecast_boundary_are_assigned_exactly_once():
    b = observations()
    b['batch_timestamp'][0, 7] = 1.
    p = plan(b)
    assert p.windows[-1].receivers[0, 7]
    assert p.windows[3].receivers[0, 4]  # .6 is future, not last historical interval
    assert torch.equal(sum(w.receivers.long() for w in p.windows), b['batch_pad_mask'].long())


def test_record_bounds_use_task_origin_not_first_observation(tmp_path):
    from test_tpatch_records import make_data, process
    from datasets import TPatchRecordDataset
    make_data(tmp_path, 'ushcn')
    g, data = process(tmp_path, 'ushcn')
    g.update(rolling_window_size=2 / 48, rolling_within_relations=[3, 4])
    ds = TPatchRecordDataset(data['train_data'], type='train', global_args=g, pack_attention=True)
    # Second task starts at month 1, while absolute record time is retained.
    sample = ds[1]
    torch.testing.assert_close(sample[0, 4:7], torch.tensor([1 / 48, 25 / 48, 26 / 48]))
    batch = ds.collate_fn([sample])
    p = batch['batch_attention_plan']
    assert len(p.windows) == 13
    assert p.windows[0].bounds[0, 0] == sample[0, 4]
    assert p.windows[-1].bounds[0, 0] == sample[0, 5]


def test_rolling_runner_uses_same_tag_for_every_dataset(tmp_path):
    import json, os, subprocess, sys
    fake, log = tmp_path / 'python', tmp_path / 'commands'
    fake.write_text(f'#!{sys.executable}\nimport json,os,sys\nwith open(os.environ["COMMAND_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON=str(fake), COMMAND_LOG=str(log), ROLLING_WINDOW_SIZE='.1')
    env.pop('RUN_TAG', None)
    run = subprocess.run(['bash', 'scripts/run_tpatch_oracle.sh'], cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    parents = [Path(c[c.index('--result_save_dir_path') + 1]).parent for c in calls]
    assert len(set(parents)) == 1
    assert all(c[c.index('--rolling-window-size') + 1] == '.1' for c in calls)
