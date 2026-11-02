import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from models import ExEn
from models.STRGCN.ExEn import relation_masks
from datasets import ExEnPredictionDataset
from trainers import IrrFCGraphForecastTrainer

torch.set_num_threads(1)


def graph():
    return dict(batch_value=torch.tensor([[1., 2., 3., 4., 999., 888., 5.]]),
                batch_timestamp=torch.tensor([[0., 0., 1., 1., 2., 3., 3.]]),
                batch_var_idx=torch.tensor([[0, 1, 0, 1, 1, 1, 0]]),
                batch_node_type=torch.tensor([[0, 1, 0, 1, 3, 3, 2]]),
                batch_pred_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 0]]),
                batch_pad_mask=torch.ones(1, 7))


def model(**kw):
    torch.manual_seed(7)
    return ExEn(num_vars=2, exogenous_var_indices=[0], hid_dim=8, n_layer=2,
                n_cross_layer=2, **kw)


def test_masks_match_definition():
    b = graph()
    masks = relation_masks(b['batch_timestamp'], b['batch_var_idx'],
                           b['batch_node_type'], b['batch_pad_mask'])
    for r, mask in enumerate(masks):
        for i in range(7):
            for j in range(7):
                t, v, ty = b['batch_timestamp'][0], b['batch_var_idx'][0], b['batch_node_type'][0]
                causal = t[j] <= t[i] and ty[j] != 3
                xi, xj = ty[i] in (0, 2), ty[j] in (0, 2)
                expected = causal and ((xi and xj and v[i] == v[j]) if r == 0 else
                                       (not xi and not xj and v[i] == v[j]) if r == 1 else
                                       (not xi and xj))
                assert bool(mask[0, i, j]) == bool(expected)
    assert not torch.stack(masks)[..., 4:6].any()  # queries never send


@pytest.mark.parametrize('embedding', ['rbf', 'rope', 'transformer'])
@pytest.mark.parametrize('batch_size', [1, 2])
def test_forward_backward(embedding, batch_size):
    b = {k: v.repeat(batch_size, 1) for k, v in graph().items()}
    m = model(time_embedding_type=embedding)
    b['batch_value'].requires_grad_()
    out = m(**b)
    assert out.shape == (batch_size, 7, 1) and torch.isfinite(out).all()
    out[:, 4:6].square().mean().backward()
    assert torch.equal(b['batch_value'].grad[:, 4:6], torch.zeros(batch_size, 2))
    assert b['batch_value'].grad[:, :4].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.cross_layers.parameters())


def test_labels_nan_inf_never_enter_encoder_or_gradients():
    m, b = model().eval(), graph()
    expected = m(**b)
    for value in [float('nan'), float('inf'), -1e9]:
        b['batch_value'][:, 4:6] = value
        torch.testing.assert_close(m(**b), expected)
    b['batch_value'].requires_grad_()
    m(**b)[:, 4:6].sum().backward()
    assert torch.isfinite(m.encoder.weight.grad).all()


def test_late_future_exo_cannot_affect_early_query_multilayer():
    m, b = model().eval(), graph()
    expected = m(**b)
    b['batch_value'][0, 6] += 100
    changed = m(**b)
    torch.testing.assert_close(changed[:, :5], expected[:, :5])
    assert not torch.allclose(changed[:, 5], expected[:, 5])


def test_padding_nonfinite_and_isolated_query():
    m, b = model().eval(), graph()
    expected = m(**b)
    for k, v in b.items():
        fill = -999 if k in ('batch_var_idx', 'batch_node_type') else float('nan')
        if k in ('batch_pad_mask', 'batch_pred_mask'):
            fill = 0
        b[k] = torch.cat((v, torch.full((1, 2), fill, dtype=v.dtype)), dim=1)
    torch.testing.assert_close(m(**b)[:, :7], expected)
    assert (m(**b)[:, 7:] == 0).all()
    # A query with no neighbors must be finite and independent of its placeholder.
    single = {k: v[:, 4:5] for k, v in graph().items()}
    assert torch.isfinite(m(**single)).all()


def test_other_endogenous_variable_has_no_message_path():
    m = ExEn(num_vars=3, exogenous_var_indices=[0], hid_dim=8, n_layer=3)
    m.eval()
    b = graph()
    b['batch_var_idx'][0, 1] = 2  # second endogenous variable
    before = m(**b)
    b['batch_value'][0, 1] += 1e4
    torch.testing.assert_close(m(**b)[:, 4:6], before[:, 4:6])


def data():
    return np.array([[s, v, t, (s + 1) * .1 + t * (v + 1)]
                     for s in range(4) for t in [0., .4, 1.1, 1.8, 2.2, 2.8]
                     for v in [10, 20, 30]], dtype=float)


def dataset(raw=None, global_args=None, split='train', known=(10,)):
    return ExEnPredictionDataset(data() if raw is None else raw, type=split,
                                history_len=2, forecast_len=1, time_end=3,
                                exogenous_var_ids=[10, 30], known_future_exogenous_var_ids=known,
                                global_args={} if global_args is None else global_args)


def test_dataset_roles_and_unavailable_future_exo():
    ds = dataset()
    b = ds.collate_fn([ds[0], ds[1]])
    assert set(b['batch_node_type'][b['batch_pad_mask'].bool()].tolist()) == {0, 1, 2, 3}
    future = b['batch_timestamp'] >= 2 / 3
    assert not b['batch_pad_mask'][future & (b['batch_var_idx'] == 2)].any()
    assert (b['batch_pred_mask'].bool() == ((b['batch_node_type'] == 3) & b['batch_pad_mask'].bool())).all()
    assert ds.global_args['exogenous_var_indices'] == [0, 2]


def test_split_mapping_is_shared_even_with_missing_variable():
    g = {}
    train = dataset(global_args=g)
    raw = data()
    valid = dataset(raw[raw[:, 1] != 10], g, 'valid')
    assert valid.var2col == train.var2col == {10: 0, 20: 1, 30: 2}
    assert g['num_of_vals'] == 3


def test_missing_future_exo_supported():
    ds = dataset(known=())
    b = ds.collate_fn([ds[0]])
    assert not ((b['batch_node_type'] == 2) & b['batch_pad_mask'].bool()).any()
    m = ExEn(global_args=ds.global_args, hid_dim=8)
    assert torch.isfinite(m(**b)).all()


def test_train_evaluate_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(123)
    g = {'device': 'cpu'}
    ds = dataset(global_args=g)
    loader = DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)
    m = ExEn(global_args=g, hid_dim=8)
    opt = torch.optim.Adam(m.parameters(), lr=.01)
    trainer = IrrFCGraphForecastTrainer(m, opt, None, None, str(tmp_path), str(tmp_path),
                                       max_epoch_num=1, do_inverse_scaling=False, global_args=g)
    b = next(iter(loader))
    mask = b['batch_pred_mask'].bool()
    with torch.no_grad():
        initial = ((m(**b).squeeze(-1)[mask] - b['batch_value'][mask]) ** 2).mean().item()
    for _ in range(25):
        assert np.isfinite(trainer.train_one_epoch(loader))
    result = trainer.evaluate(loader, ['masked_mse_long', 'masked_mae_long'])
    assert np.isfinite(result[0]) and result[0] < initial
    torch.save(m.state_dict(), tmp_path / 'exen.pt')
    restored = ExEn(global_args=g, hid_dim=8)
    restored.load_state_dict(torch.load(tmp_path / 'exen.pt', weights_only=True))
    m.eval(); restored.eval()
    torch.testing.assert_close(m(**b), restored(**b))


def test_validation_errors():
    m, b = model(), graph()
    with pytest.raises(ValueError, match='required'):
        m(**{k: v for k, v in b.items() if k != 'batch_node_type'})
    b['batch_pred_mask'][0, 0] = 1
    with pytest.raises(ValueError, match='Prediction mask'):
        m(**b)
    with pytest.raises(ValueError, match='Sandwich'):
        model(use_info_agg=True)
    with pytest.raises(ValueError, match='direct masked'):
        model(gcn_type='cheb', gcn_depth=2)


def test_cli_pipeline_and_train_only_scaler(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    raw = data()
    held_out = raw.copy()
    held_out[:, 3] += 100  # must not affect training scaling statistics
    archive = tmp_path / 'splits.npz'
    np.savez(archive, train_data=raw, valid_data=held_out, test_data=held_out)
    config = yaml.safe_load((root / 'config/ExEn/ExEn_npz.yaml').read_text())
    config['global_config']['exen_data_path'] = str(archive)
    config['train_config']['trainer_params']['max_epoch_num'] = 1
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    run = subprocess.run([sys.executable, 'main.py', '--config_path', str(path),
                          '--model_save_dir_path', str(tmp_path / 'model'),
                          '--result_save_dir_path', str(tmp_path / 'result')],
                         cwd=root, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (tmp_path / 'result' / 'result.json').exists()
    assert (tmp_path / 'result' / 'test_y_pred.npy').exists()
    with np.load(tmp_path / 'model' / 'scaler.npz', allow_pickle=True) as saved:
        maxima = saved['max_vals'].item()
    for var in [10, 20, 30]:
        assert maxima[var] == raw[raw[:, 1] == var, 3].max()


def test_batch_and_query_order_do_not_change_predictions():
    m, b = model().eval(), graph()
    expected = m(**b)
    perm = torch.tensor([6, 3, 5, 0, 2, 4, 1])
    permuted = {k: v[:, perm] for k, v in b.items()}
    torch.testing.assert_close(m(**permuted), expected[:, perm])
    together = {k: torch.cat([v, v.clone()], dim=0) for k, v in b.items()}
    together['batch_value'][1, 0] += 1e4
    torch.testing.assert_close(m(**together)[:1], expected)


def test_loss_ignores_nontarget_nan():
    # Selecting targets must precede subtraction: NaN * 0 is still NaN.
    pred = torch.tensor([1., 3.], requires_grad=True)
    truth = torch.tensor([float('nan'), 2.])
    loss = IrrFCGraphForecastTrainer.loss_func(None, pred, truth, mask=torch.tensor([0, 1]))
    assert loss.item() == 1
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([0., 2.]))
