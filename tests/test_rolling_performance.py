"""Compact chronological execution must preserve full-tensor values/gradients."""
import copy

import pytest
import torch

from models import ExEnGraphTransformer
from models.STRGCN.RollingWindows import pack_rolling_attention_plan
from test_within_variable_attention import observations


def make_model(**kwargs):
    return ExEnGraphTransformer(num_vars=4, exogenous_var_indices=[0, 1], hid_dim=8,
        num_heads=2, n_layer=2, n_variable_layer=1, n_cross_layer=2,
        rolling_window_size=.2, dropout=0., **kwargs)


def make_plan(batch):
    return pack_rolling_attention_plan(batch['batch_timestamp'], batch['batch_var_idx'],
        batch['batch_node_type'], batch['batch_pad_mask'].bool(), .2,
        bounds=torch.tensor([[0., .6, 1.]]).repeat(len(batch['batch_value']), 1),
        within_relations=(3, 4), receiver_chunk=2, max_score_elements=32)


@pytest.mark.parametrize('backend', ['sdpa', 'reference'])
@pytest.mark.parametrize('embedding', ['rbf', 'rope', 'transformer'])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_compact_matches_original_outputs_and_all_gradients(backend, embedding, checkpoint):
    torch.manual_seed(72)
    a = make_model(attention_backend=backend, time_embedding_type=embedding,
                   checkpoint_attention=checkpoint)
    old = copy.deepcopy(a)
    old._run_rolling_windows = old._run_rolling_reference
    b = {k: v.repeat(3, 1) for k, v in observations().items()}
    b['batch_pad_mask'][1, [0, 1, 4, 5]] = 0  # no X; empty windows
    b['batch_pad_mask'][2] = 0  # whole record padded
    p = make_plan(b)
    x, y = a(**b, batch_attention_plan=p), old(**b, batch_attention_plan=p)
    torch.testing.assert_close(x, y, atol=4e-6, rtol=4e-5)
    mask = b['batch_pred_mask'].bool() & b['batch_pad_mask'].bool()
    x.squeeze(-1)[mask].square().sum().backward()
    y.squeeze(-1)[mask].square().sum().backward()
    for (name, v), (_, w) in zip(a.named_parameters(), old.named_parameters()):
        vg = torch.zeros_like(v) if v.grad is None else v.grad
        wg = torch.zeros_like(w) if w.grad is None else w.grad
        torch.testing.assert_close(vg, wg, atol=8e-5, rtol=8e-4, msg=name)


@pytest.mark.parametrize('time_bias,variable_bias', [(False, False), (True, False), (False, True)])
def test_optional_biases_preserve_original_results(time_bias, variable_bias):
    a = make_model(use_time_bias=time_bias, use_variable_bias=variable_bias)
    old = copy.deepcopy(a); old._run_rolling_windows = old._run_rolling_reference
    b = observations(); p = make_plan(b)
    torch.testing.assert_close(a(**b, batch_attention_plan=p), old(**b, batch_attention_plan=p))


def test_node_work_is_current_only_and_time_embeddings_are_once_per_layer():
    a = make_model(checkpoint_attention=False)
    old = copy.deepcopy(a); old._run_rolling_windows = old._run_rolling_reference
    b = observations(); p = make_plan(b)
    def count(model):
        stats = dict(q=0, ffn=0, time=0, readout=0, archive=0)
        handles = []
        def rows(key):
            def hook(module, args):
                stats[key] += args[0].numel() // args[0].shape[-1]
            return hook
        def time_hook(module, args): stats['time'] += 1
        for layer in [*model.temporal_layers, *model.variable_layers, *model.cross_layers]:
            for projection in layer.q_proj:
                handles.append(projection.register_forward_pre_hook(rows('q')))
            handles.append(layer.ffn[0].register_forward_pre_hook(rows('ffn')))
            handles.append(layer.time_embedding.register_forward_pre_hook(time_hook))
        handles.append(model.window_memory.interactions.register_forward_pre_hook(rows('readout')))
        out = model(**b, batch_attention_plan=p)
        for h in handles: h.remove()
        return stats, out
    current, x = count(a); reference, y = count(old)
    torch.testing.assert_close(x, y)
    assert current['q'] * len(p.windows) == reference['q']
    assert current['ffn'] * len(p.windows) == reference['ffn']
    assert current['time'] * len(p.windows) == reference['time']
    assert current['readout'] == int((b['batch_node_type'] != 3).sum())
    assert current['readout'] < reference['readout']


def test_cache_does_not_persist_between_forward_calls_or_optimizer_steps():
    m = make_model(checkpoint_attention=True)
    old = copy.deepcopy(m); old._run_rolling_windows = old._run_rolling_reference
    b = observations(); p = make_plan(b)
    opt = torch.optim.Adam(m.parameters(), lr=.001)
    old_opt = torch.optim.Adam(old.parameters(), lr=.001)
    for step in range(2):
        opt.zero_grad(); old_opt.zero_grad()
        x, y = m(**b, batch_attention_plan=p), old(**b, batch_attention_plan=p)
        torch.testing.assert_close(x, y, atol=4e-6, rtol=4e-5)
        x[:, 6:].square().sum().backward(); y[:, 6:].square().sum().backward()
        opt.step(); old_opt.step()


def test_optimizer_state_matches_when_next_batch_has_no_known_observations():
    m = make_model(checkpoint_attention=False)
    old = copy.deepcopy(m); old._run_rolling_windows = old._run_rolling_reference
    optimizer = torch.optim.SGD(m.parameters(), lr=.001, momentum=.9, weight_decay=.01)
    reference_optimizer = torch.optim.SGD(old.parameters(), lr=.001, momentum=.9, weight_decay=.01)
    for empty in (False, True):
        b = observations()
        if empty:
            b['batch_pad_mask'][0, :6] = 0
        p = make_plan(b)
        for network, opt in ((m, optimizer), (old, reference_optimizer)):
            opt.zero_grad(set_to_none=True)
            network(**b, batch_attention_plan=p)[:, 6:].square().sum().backward()
            opt.step()
        for (name, parameter), (_, reference) in zip(m.named_parameters(), old.named_parameters()):
            torch.testing.assert_close(parameter, reference, atol=3e-6, rtol=3e-5, msg=name)
