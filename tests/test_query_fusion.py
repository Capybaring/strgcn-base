"""Information-flow, interface parity, transfer and both real encoder tests."""

import copy
import json
import subprocess
import sys

import pytest
import torch
from test_tpatch_records import make_data

from data_processors import TPatchRecordProcessor
from datasets import QueryFusionDataset
from models.exogenous.adapter import HiPatchQueryFusion, TPatchQueryFusion
from models.exogenous.fusion import DynamicLagFusion, Observations
from utils.scaler import TPatchInputScaler


def prepared(root, dataset="activity", mode="FullExo"):
    g = {"device": "cpu"}
    data = TPatchRecordProcessor(
        g, dataset, data_root=root, max_samples_per_split=2
    ).process()
    scale = TPatchInputScaler()
    scale.fit(data["seen_data"])
    ds = QueryFusionDataset(
        scale.transform(data["train_data"]),
        type="train",
        global_args=g,
        patch_size=dict(mimic=8, physionet=8, activity=300, ushcn=2)[dataset],
        use_exogenous=mode != "NoExo",
        known_future_exogenous_var_ids=[] if mode == "HistExo" else "all",
    )
    return g, ds


@pytest.mark.parametrize("cls", [TPatchQueryFusion, HiPatchQueryFusion])
@pytest.mark.parametrize("dataset", ["mimic", "physionet", "activity", "ushcn"])
@pytest.mark.parametrize("mode", ["none", "mlp", "cross", "v1", "v2"])
def test_adapters_gradients_and_original_decoder_parity(tmp_path, cls, dataset, mode):
    make_data(tmp_path, dataset)
    g, ds = prepared(tmp_path, dataset)
    b = ds.collate_fn([ds[0], ds[1]])
    m = cls(
        g,
        hid_dim=8,
        n_layer=1,
        n_heads=1,
        plugin_hidden=8,
        plugin_heads=2,
        fusion_mode=mode,
    ).eval()
    assert not b["batch_backbone_mask"][..., len(ds.packet["target_ids"]) :].any()
    features = m.encode_queries(b)
    original = m.core.forecasting(
        b["batch_backbone_queries"],
        b["batch_backbone_values"],
        b["batch_backbone_times"],
        b["batch_backbone_mask"],
    )[0, ..., : len(ds.packet["target_ids"])]
    torch.testing.assert_close(m.decode(features), original)
    pred = m(**b)
    assert torch.isfinite(pred).all()
    pred[b["batch_pred_mask"].bool()].square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters()
    )
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    if mode not in ("none", "mlp"):
        assert m.fusion.out.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("cls", [TPatchQueryFusion, HiPatchQueryFusion])
@pytest.mark.parametrize("mode", ["cross", "v1", "v2"])
@pytest.mark.parametrize("inputs", ["FullExo", "HistExo", "NoExo"])
def test_future_label_late_x_and_noexo_isolation(tmp_path, cls, mode, inputs):
    make_data(tmp_path, "activity")
    g, ds = prepared(tmp_path, mode=inputs)
    sample, origin = ds[0]
    b = ds.collate_fn([(sample, origin)])
    m = cls(
        g,
        hid_dim=8,
        n_layer=1,
        n_heads=1,
        plugin_hidden=8,
        plugin_heads=2,
        fusion_mode=mode,
    ).eval()
    before = m(**b)
    altered = sample.clone()
    q = b["batch_pred_mask"][0].bool()
    altered[q, 1] = 1e8
    torch.testing.assert_close(before, m(**ds.collate_fn([(altered, origin)])))
    ix = torch.where((b["batch_node_type"][0] == 2) & b["batch_pad_mask"][0].bool())[0]
    if len(ix):
        late = ix[b["batch_timestamp"][0, ix].argmax()]
        cut = b["batch_timestamp"][0, late]
        altered = sample.clone()
        altered[late, 1] += 100
        earlier = b["batch_pred_mask"].bool() & (b["batch_timestamp"] < cut)
        assert earlier.any()
        torch.testing.assert_close(
            before[earlier], m(**ds.collate_fn([(altered, origin)]))[earlier]
        )
    if inputs == "NoExo":
        f = m.encode_queries(b)
        torch.testing.assert_close(m.fuse(f, b), f)


def toy():
    torch.manual_seed(3)
    x = Observations(
        torch.tensor([[1.0, 4.0, 2.0]]),
        torch.tensor([[0.1, 0.3, 0.6]]),
        torch.tensor([[0, 0, 0]]),
        torch.ones(1, 3, dtype=torch.bool),
    )
    y = Observations(
        torch.tensor([[2.0, -1.0]]),
        torch.tensor([[0.4, 0.7]]),
        torch.tensor([[1, 1]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    return x, y


def test_transfer_is_dynamic_and_chunk_order_mask_invariant():
    x, y = toy()
    m = DynamicLagFusion(6, 2, hidden=8, heads=2, mode="v2", chunk_size=1).eval()
    f = torch.randn(1, 2, 6)
    t = torch.tensor([[0.8, 1.0]])
    ids = torch.ones(1, 2, dtype=torch.long)
    valid = torch.ones_like(ids).bool()
    before = m(f, t, ids, valid, x, y, x)
    changed = copy.deepcopy(y)
    changed.values += 20
    after = m(f, t, ids, valid, x, changed, x)
    assert not torch.allclose(before, after)
    state, support = m.encode_relations(y, x)
    assert support[0, 0, 1, 0]
    torch.testing.assert_close(state[0, :, 1].sum(-1), torch.ones(2))
    m.chunk_size = 100
    torch.testing.assert_close(before, m(f, t, ids, valid, x, y, x))
    order = torch.tensor([2, 0, 1])
    permuted = Observations(
        *(getattr(x, k)[:, order] for k in ("values", "times", "variables", "valid"))
    )
    torch.testing.assert_close(before, m(f, t, ids, valid, permuted, y, permuted))
    padded = Observations(
        torch.cat([x.values, torch.tensor([[float("nan")]])], 1),
        torch.cat([x.times, torch.tensor([[float("nan")]])], 1),
        torch.cat([x.variables, torch.tensor([[999]])], 1),
        torch.tensor([[1, 1, 1, 0]]).bool(),
    )
    torch.testing.assert_close(before, m(f, t, ids, valid, padded, y, x))
    before.square().sum().backward()
    assert m.transfer_scale.grad.abs().sum() > 0


def test_v1_has_no_history_transfer_and_empty_is_identity():
    x, y = toy()
    m = DynamicLagFusion(6, 2, hidden=8, heads=2, mode="v1")
    f = torch.randn(1, 2, 6)
    t = torch.tensor([[0.8, 1.0]])
    ids = torch.ones(1, 2, dtype=torch.long)
    v = torch.ones_like(ids).bool()
    a = m(f, t, ids, v, x)
    y.values += 999
    torch.testing.assert_close(a, m(f, t, ids, v, x, y, x))
    x.valid[:] = False
    torch.testing.assert_close(f, m(f, t, ids, v, x))


def test_exen_xy_formula_and_source_gradients():
    x, _ = toy()
    m = DynamicLagFusion(6, 2, hidden=8, heads=2, mode="v1").double()
    x.values = x.values.double().requires_grad_()
    x.times = x.times.double()
    f = torch.randn(1, 2, 6, dtype=torch.double, requires_grad=True)
    times = torch.tensor([[0.2, 0.8]], dtype=torch.double)
    ids = torch.ones(1, 2, dtype=torch.long)
    valid = ids.bool()
    # Independent direct formula: content / sqrt(dh) + time dot + variable dot.
    h = (
        m.value_enc(x.values[..., None])
        + m.variable(x.variables)
        + torch.sin(m.time(x.times[..., None]))
    )
    head = lambda a: a.reshape(1, -1, 2, 4).transpose(1, 2)
    q, k, v = head(m.q(m.query(f))), head(m.k(h)), head(m.v(h))
    pt, ps = (
        head(torch.sin(m.time(times[..., None]))),
        head(torch.sin(m.time(x.times[..., None]))),
    )
    score = q @ k.transpose(-1, -2) / 2 + m.time_scale[None, :, None, None] * (
        pt @ ps.transpose(-1, -2)
    )
    score = score + m.var_scale[None, :, None, None] * (
        head(m.u(ids)) @ head(m.w(x.variables)).transpose(-1, -2)
    )
    legal = x.times[:, None, :] <= times[:, :, None]
    weight = score.masked_fill(~legal[:, None], -torch.inf).softmax(-1)
    expected = f + m.out((weight @ v).transpose(1, 2).reshape(1, 2, 8))
    actual = m(f, times, ids, valid, x)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (f, x.values), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (f, x.values))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


def test_cli_matrix_exports_and_refuses_existing_output(tmp_path):
    from pathlib import Path

    make_data(tmp_path / "data", "activity")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "runs"
    cmd = [
        sys.executable,
        "scripts/run_query_fusion.py",
        "--output",
        str(output),
        "--data-root",
        str(tmp_path / "data"),
        "--datasets",
        "activity",
        "--variants",
        "v1",
        "v2",
        "--seeds",
        "3407",
        "--epochs",
        "1",
        "--max-samples-per-split",
        "2",
        "--small-model",
        "--stream-logs",
    ]
    run = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Start training." in run.stdout
    rows = json.loads((output / "summary.json").read_text())
    assert len(rows) == 4
    assert all(
        row["exit_code"] == 0 and row["parameters"] > 0 and row["train_seconds"] > 0
        for row in rows
    )
    for path in output.glob("*/activity/3407/*/results"):
        assert "Start training." in (path.parent / "run.log").read_text()
        metrics = json.loads((path / "metrics.json").read_text())
        assert metrics["protocol"]["fusion"]["mode"] in ("v1", "v2")
        for name in (
            "test_y_mask.npy",
            "test_var_idx.npy",
            "test_record_id.npy",
            "test_window_id.npy",
        ):
            assert (path / name).exists()
    again = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=10)
    assert again.returncode != 0 and "FileExistsError" in again.stderr
