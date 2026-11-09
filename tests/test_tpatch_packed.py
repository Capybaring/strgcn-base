"""Packed/loop equivalence and independent t-PatchGNN metric reference."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from models import ExEnGraphTransformer
from models.STRGCN.PackedAttention import pack_attention_plan
from utils.tpatch_evaluation import variable_error_sums, variable_macro_loss, summarize_variable_errors
from utils.scaler import TPatchInputScaler
from test_exen import graph


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('embedding', ['rbf', 'rope', 'transformer'])
@pytest.mark.parametrize('backend', ['sdpa', 'reference'])
def test_packed_matches_loop_outputs_and_gradients(checkpoint, embedding, backend):
    torch.manual_seed(17)
    options = dict(num_vars=2, exogenous_var_indices=[0], hid_dim=8, n_layer=2,
                   n_cross_layer=1, num_heads=2, dropout=0., attention_backend=backend,
                   time_embedding_type=embedding, checkpoint_attention=checkpoint)
    a = ExEnGraphTransformer(**options); b = copy.deepcopy(a)
    sample = {k: v.repeat(3, 1) for k, v in graph().items()}
    sample['batch_pad_mask'][1, 0] = 0
    # Exercise a record with no exogenous sources and padded receiver rows.
    sample['batch_pad_mask'][2, [0, 2, 6]] = 0
    plan = pack_attention_plan(sample['batch_timestamp'], sample['batch_var_idx'],
                              sample['batch_node_type'], sample['batch_pad_mask'].bool(),
                              receiver_chunk=2, max_score_elements=16).to('cpu')
    out_a = a(**sample)
    out_b = b(**sample, batch_attention_plan=plan)
    torch.testing.assert_close(out_a, out_b, atol=3e-6, rtol=3e-5)
    out_a.square().sum().backward(); out_b.square().sum().backward()
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other
        if p.grad is None:
            assert q.grad is None or (q.grad == 0).all(), name
        else:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=5e-5, rtol=5e-4, msg=name)


def test_packed_query_values_and_future_causal_edges_cannot_leak():
    torch.manual_seed(8)
    model = ExEnGraphTransformer(num_vars=2, exogenous_var_indices=[0], hid_dim=8,
                                 num_heads=2, dropout=0., checkpoint_attention=False).eval()
    b = graph()
    plan = pack_attention_plan(b['batch_timestamp'], b['batch_var_idx'], b['batch_node_type'], b['batch_pad_mask'].bool())
    before = model(**b, batch_attention_plan=plan)
    b['batch_value'][:, 4:6] = torch.nan
    torch.testing.assert_close(before, model(**b, batch_attention_plan=plan))
    b['batch_value'][0, 6] += 100
    after = model(**b, batch_attention_plan=plan)
    torch.testing.assert_close(before[:, :5], after[:, :5])
    assert not torch.allclose(before[:, 5], after[:, 5])


def test_variable_macro_metrics_match_dense_reference_and_are_batch_invariant():
    # Sparse variable 1 must have the same variable weight as frequent variable 0.
    truth = torch.zeros(2, 3, 3, dtype=torch.float64)
    pred = torch.tensor([[[1., 2., 99.], [3., 99., 99.], [10., 99., 99.]],
                         [[2., 99., 99.], [99., 99., 99.], [99., 99., 99.]]], dtype=torch.float64)
    mask = torch.zeros_like(truth, dtype=torch.bool)
    mask[0, :, 0] = True; mask[0, 0, 1] = True; mask[1, 0, 0] = True
    ids = torch.arange(3).expand(2, 3, 3)
    se = ((truth - pred).square() * mask).sum((0, 1))
    ae = ((truth - pred).abs() * mask).sum((0, 1))
    count = mask.sum((0, 1))
    expected_mse = (se / (count + 1e-8)).sum() / torch.count_nonzero(count)
    expected_mae = (ae / (count + 1e-8)).sum() / torch.count_nonzero(count)
    flat = lambda x: x.reshape(2, -1)
    stats = variable_error_sums(flat(pred), flat(truth), flat(mask), flat(ids), 3)
    report = summarize_variable_errors(*stats)
    np.testing.assert_allclose(report['mse'], expected_mse)
    np.testing.assert_allclose(report['mae'], expected_mae)
    np.testing.assert_allclose(report['rmse'], expected_mse.sqrt())
    pieces = [variable_error_sums(x.reshape(1, -1), y.reshape(1, -1), m.reshape(1, -1), v.reshape(1, -1), 3)
              for x, y, m, v in zip(pred, truth, mask, ids)]
    combined = [sum(piece[j] for piece in pieces) for j in range(3)]
    assert summarize_variable_errors(*combined) == report
    p = flat(pred).clone().requires_grad_()
    loss = variable_macro_loss(p, flat(truth), flat(mask), flat(ids), 3)
    np.testing.assert_allclose(loss.detach(), expected_mse)
    loss.backward()
    assert torch.isfinite(p.grad).all() and (p.grad[~flat(mask)] == 0).all()


def test_upstream_scaling_seen_scope_and_ushcn_identity():
    def packet(dataset, values):
        return dict(kind='tpatch_records', dataset=dataset, names=['a', 'b'],
                    records=[('id', np.array([0., 1.]), np.array(values), np.ones((2, 2), dtype=bool))])
    p = packet('activity', [[2., 5.], [10., 5.]])
    s = TPatchInputScaler(); s.fit(p)
    np.testing.assert_allclose(s.minimum, [2., 5.]); np.testing.assert_allclose(s.ranges, [8., 1e-8])
    p['dataset'] = 'ushcn'; s = TPatchInputScaler(); s.fit(p)
    np.testing.assert_array_equal(s.minimum, [0., 0.]); np.testing.assert_array_equal(s.ranges, [1., 1.])


def test_all_tpatch_configs_use_official_batches_and_new_metrics():
    root = Path(__file__).resolve().parents[1]
    for path in root.glob('config/*/tpatch/*.yaml'):
        c = yaml.safe_load(path.read_text())
        assert c['train_config']['batch_size'] == (192 if path.stem == 'ushcn' else 32)
        assert c['train_config']['trainer_name'] == 'TPatchForecastTrainer'
        assert c['train_config']['trainer_params']['eval_metric_key'] == 'mse'
        assert c['train_config']['metrics'] == ['mse', 'mae', 'rmse']
        assert c['data_config']['scaler_name'] == 'TPatchInputScaler'
        if c['model_config']['model_name'] == 'ExEnGraphTransformer':
            assert c['data_config']['dataset_params']['pack_attention']
            assert not c['model_config']['model_params']['checkpoint_attention']


def test_random_ragged_packed_tied_times_and_empty_groups():
    torch.manual_seed(71)
    B, L, V = 3, 57, 7
    ids = torch.randint(V, (B, L))
    times = torch.cat([torch.randint(0, 5, (B, 30)) / 10,
                       torch.randint(6, 10, (B, L - 30)) / 10], dim=1)
    valid = torch.rand(B, L) > .15
    exo = ids < 3
    future = times >= .6
    role = torch.where(exo, torch.where(future, 2, 0), torch.where(future, 3, 1))
    b = dict(batch_value=torch.randn(B, L), batch_timestamp=times, batch_var_idx=ids,
             batch_node_type=role, batch_pad_mask=valid.float(),
             batch_pred_mask=(future & ~exo & valid).float())
    options = dict(num_vars=V, exogenous_var_indices=[0, 1, 2], hid_dim=8,
                   num_heads=2, n_layer=1, n_cross_layer=1, dropout=0., checkpoint_attention=False)
    a = ExEnGraphTransformer(**options); c = copy.deepcopy(a)
    plan = pack_attention_plan(times, ids, role, valid, receiver_chunk=5, max_score_elements=128)
    x = a(**b); y = c(**b, batch_attention_plan=plan)
    torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-5)
    x.sum().backward(); y.sum().backward()
    for (n, p), (_, q) in zip(a.named_parameters(), c.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=5e-5, rtol=5e-4, msg=n)
