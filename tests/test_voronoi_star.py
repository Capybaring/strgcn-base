"""Analytic geometry, calibrated fields, sparse attention and adapter isolation."""

import copy
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from models.exogenous.fusion import DynamicLagFusion, Observations
from models.exogenous.voronoi_star import (
    VoronoiStarTransformer,
    calibrate_voronoi_references,
    relative_density_field,
    voronoi_star,
)

ROOT = Path(__file__).resolve().parents[1]


def example():
    times = torch.tensor(
        [[2.0, 8.0, 12.0, 0.0, 4.0, 9.0, 20.0, 1.0, 7.0, 11.0, 15.0]],
        dtype=torch.double,
    )
    ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]])
    return Observations(
        torch.linspace(-1, 1, 11, dtype=torch.double)[None],
        times,
        ids,
        torch.ones_like(ids, dtype=torch.bool),
    )


def core(**kwargs):
    m = VoronoiStarTransformer(8, 3, heads=2, node_dim=4, **kwargs).double()
    m.fit_reference([example()])
    return m


def test_receiver_A8_connects_only_nearest_groups_and_uses_projected_cells():
    x = example()
    g = voronoi_star(x.times, x.variables, x.valid, 3)
    sources = g.neighbors[0, 1, g.edge_valid[0, 1]].tolist()
    assert sorted(sources) == [0, 1, 2, 4, 5, 8, 9]
    torch.testing.assert_close(
        g.log_rho[0, 1].exp(), torch.tensor([0.2, 0.4, 0.5], dtype=torch.double)
    )
    # Distances/columns depend on category equality, never the ID magnitude.
    perm = torch.tensor([2, 0, 1])
    other = voronoi_star(x.times, perm[x.variables], x.valid, 3)
    torch.testing.assert_close(other.log_rho[:, :, perm], g.log_rho)
    assert sorted(other.neighbors[0, 1, other.edge_valid[0, 1]].tolist()) == sorted(
        sources
    )


def test_single_variable_is_standard_voronoi_and_one_side_is_mirrored():
    t = torch.tensor([[0.0, 1.0, 4.0, 6.0]], dtype=torch.double)
    g = voronoi_star(
        t,
        torch.zeros_like(t, dtype=torch.long),
        torch.ones_like(t, dtype=torch.bool),
        1,
    )
    torch.testing.assert_close(
        g.log_rho[0, :, 0].exp(), t.new_tensor([1.0, 0.5, 0.4, 0.5])
    )
    singleton = voronoi_star(
        t[:, :1],
        torch.zeros((1, 1), dtype=torch.long),
        torch.ones((1, 1), dtype=torch.bool),
        1,
    )
    assert not singleton.known.any()
    assert singleton.edge_valid.sum() == 1  # self, with unknown density


def test_ties_retain_all_values_and_never_create_infinite_density():
    t = torch.tensor([[0.0, 1.0, 1.0, 4.0, 1.0]], dtype=torch.double)
    ids = torch.tensor([[0, 0, 0, 0, 1]])
    g = voronoi_star(t, ids, torch.ones_like(ids, dtype=torch.bool), 2)
    assert torch.isfinite(g.log_rho).all()
    torch.testing.assert_close(g.log_rho[0, 1], g.log_rho[0, 2])
    assert g.log_rho[0, 1, 0].exp() == 0.5
    assert not g.known[0, 1, 1]
    assert sorted(g.neighbors[0, 0, g.edge_valid[0, 0]].tolist()) == [0, 1, 2, 4]
    # Reflections swap direction but preserve temporal cells.
    reverse = voronoi_star(10 - t, ids, torch.ones_like(ids, dtype=torch.bool), 2)
    torch.testing.assert_close(g.log_rho, reverse.log_rho)
    assert torch.equal(g.known, reverse.known)


def test_density_share_global_level_and_unknown_mask_match_equations():
    logrho = torch.tensor(
        [[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0]]], dtype=torch.double
    ).log()
    known = torch.ones_like(logrho, dtype=torch.bool)
    z, mask = relative_density_field(
        logrho,
        known,
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((3, 3)),
        torch.ones((3, 3), dtype=torch.long),
    )
    torch.testing.assert_close(
        z[0, 0, :, 1], torch.tensor([0.75, 1.5, 0.75], dtype=torch.double).log()
    )
    torch.testing.assert_close(
        z[0, 0, :, 2], torch.full((3,), math.log(1.2), dtype=torch.double)
    )
    torch.testing.assert_close(z[0, 1, :, 1], z[0, 0, :, 1])
    torch.testing.assert_close(
        z[0, 1, :, 2] - z[0, 0, :, 2], torch.full((3,), math.log(2), dtype=torch.double)
    )
    z, mask = relative_density_field(
        logrho,
        known,
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((3, 3)),
        torch.zeros((3, 3), dtype=torch.long),
    )
    assert not mask.any() and not z.any() and torch.isfinite(z).all()


def test_training_reference_is_conditional_arithmetic_density_and_checkpointed():
    x = example()
    m = core().eval()
    star = voronoi_star(x.times, x.variables, x.valid, 3)
    for a in range(3):
        for c in range(3):
            mask = (x.variables[0] == a) & star.known[0, :, c]
            expected = star.log_rho[0, mask, c].exp().mean()
            torch.testing.assert_close(m.log_reference[a, c].exp(), expected)
    original = {k: v.clone() for k, v in m.state_dict().items()}
    altered = copy.deepcopy(x)
    altered.times = 7 * x.times + 100
    m(altered)
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in original.items())
    restored = VoronoiStarTransformer(8, 3, heads=2, node_dim=4).double().eval()
    restored.load_state_dict(original)
    torch.testing.assert_close(restored(x), m(x))
    unfitted = VoronoiStarTransformer(8, 3, heads=2)
    with pytest.raises(RuntimeError, match="training X"):
        unfitted(x)


def test_no_density_is_same_star_network_and_zero_gate_matches_exactly():
    x = example()
    m = core().eval()
    plain = copy.deepcopy(m)
    plain.use_density_encoding = False
    assert sum(p.numel() for p in m.parameters()) == sum(
        p.numel() for p in plain.parameters()
    )
    torch.testing.assert_close(m(x), plain(x), atol=0, rtol=0)
    with torch.no_grad():
        m.layers[0].density_gate.fill_(0.5)
        plain.layers[0].density_gate.fill_(0.5)
    assert not torch.allclose(m(x), plain(x), atol=1e-10, rtol=1e-10)
    unchanged = plain(x)
    with torch.no_grad():
        plain.layers[0].edge_net[-1].weight.fill_(1000.0)
    torch.testing.assert_close(unchanged, plain(x), atol=0, rtol=0)


def test_zero_gate_agrees_with_independent_dense_masked_scalar_attention():
    x = example()
    m = core().eval()
    layer = m.layers[0]
    from models.STRGCN.STRGCNDirect import time_features

    p = time_features(m.time_embedding, x.times, x.valid)
    h = m.encoder(x.values[..., None]) + m.variable(x.variables) + p
    split = lambda t: t.reshape(1, 11, 2, -1).transpose(1, 2)
    q, k, v = (split(f(layer.norm1(h))) for f in (layer.q, layer.k, layer.v))
    score = q @ k.transpose(-1, -2) / math.sqrt(4)
    score = score + layer.time_scale[None, :, None, None] * (
        split(p) @ split(p).transpose(-1, -2)
    )
    u, w = (split(f(x.variables)) for f in (m.relation_q, m.relation_k))
    score = score + layer.var_scale[None, :, None, None] * (
        u @ w.transpose(-1, -2)
    ) / math.sqrt(4)
    g = voronoi_star(x.times, x.variables, x.valid, 3)
    mask = torch.zeros((11, 11), dtype=torch.bool)
    for i in range(11):
        mask[i, g.neighbors[0, i, g.edge_valid[0, i]]] = True
    message = (
        (score.masked_fill(~mask, -torch.inf).softmax(-1) @ v)
        .transpose(1, 2)
        .reshape_as(h)
    )
    expected = h + layer.out(message)
    expected = expected + layer.ffn(layer.norm2(expected))
    torch.testing.assert_close(m(x), expected, atol=1e-10, rtol=1e-9)


def test_chunk_recomputation_gradients_gate_learning_padding_and_permutation():
    torch.manual_seed(23)
    x = example()
    x.values.requires_grad_()
    m = core(layers=2, chunk_size=2).train()
    first = m(x)
    first.square().sum().backward()
    assert m.layers[0].density_gate.grad.abs().sum() > 0
    m.zero_grad()
    x.values.grad = None
    with torch.no_grad():
        for layer in m.layers:
            layer.density_gate.fill_(0.3)
    params = [x.values] + list(m.parameters())
    out = m(x)
    grads = torch.autograd.grad(out.square().sum(), params, allow_unused=True)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    names = list(dict(m.named_parameters()))
    for name in [
        "context_net.0.weight",
        "layers.0.edge_net.0.weight",
        "layers.0.q.weight",
    ]:
        assert grads[1 + names.index(name)].abs().sum() > 0
    for layer in m.layers:
        layer.chunk_size = 100
    other = m(x)
    other_grads = torch.autograd.grad(other.square().sum(), params, allow_unused=True)
    torch.testing.assert_close(out, other)
    for a, b in zip(grads, other_grads):
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)
    m.eval()
    order = torch.tensor([5, 2, 9, 1, 8, 0, 4, 7, 3, 6, 10])
    permuted = Observations(
        *(getattr(x, k)[:, order] for k in ["values", "times", "variables", "valid"])
    )
    torch.testing.assert_close(m(permuted), m(x)[:, order])
    padded = Observations(
        torch.cat([x.values, x.values.new_full((1, 1), float("nan"))], 1),
        torch.cat([x.times, x.times.new_full((1, 1), float("nan"))], 1),
        torch.cat([x.variables, torch.tensor([[999]])], 1),
        torch.cat([x.valid, torch.tensor([[False]])], 1),
    )
    torch.testing.assert_close(m(padded)[:, :11], m(x))
    assert not m(padded)[:, 11].any()


def test_calibration_uses_training_X_only_and_preserves_rng():
    m = DynamicLagFusion(
        6, 3, hidden=8, heads=2, x_encoder="voronoi_star", allow_future_x=True
    )
    sample = {
        "batch_value": torch.tensor([1.0, 99.0, 2.0, 3.0]),
        "batch_timestamp": torch.tensor([0.0, 2.0, 4.0, 5.0]),
        "batch_var_idx": torch.tensor([0, 2, 0, 1]),
        "batch_node_type": torch.tensor([0, 1, 0, 2]),
        "batch_pred_mask": torch.tensor([False, True, False, False]),
        "batch_pad_mask": torch.ones(4, dtype=torch.bool),
    }
    loader = DataLoader([sample], batch_size=1)
    state = torch.random.get_rng_state().clone()
    calibrate_voronoi_references(m, loader)
    assert torch.equal(state, torch.random.get_rng_state())
    assert not m.x_context.reference_count[2].any()
    assert not m.x_context.reference_count[:, 2].any()
    reference = m.x_context.log_reference.clone()
    sample["batch_value"][1] = float("nan")
    sample["batch_timestamp"][1] = 1e20
    calibrate_voronoi_references(m, loader)
    torch.testing.assert_close(reference, m.x_context.log_reference, atol=0, rtol=0)


@pytest.mark.parametrize("backbone", ["tpatchgnn", "hipatch", "mtand"])
def test_adapter_forecast_head_label_isolation_and_backward(tmp_path, backbone):
    from test_mtan_query_fusion import prepared as mtan_prepared
    from test_query_fusion import prepared
    from test_tpatch_records import make_data

    from models.exogenous.adapter import HiPatchQueryFusion, TPatchQueryFusion
    from models.exogenous.mtan_adapter import MTANQueryFusion

    make_data(tmp_path, "activity")
    g, ds = (mtan_prepared if backbone == "mtand" else prepared)(tmp_path, "activity")
    sample, origin = ds[0]
    batch = ds.collate_fn([(sample, origin)])
    params = {
        "hid_dim": 8,
        "n_heads": 1,
        "plugin_hidden": 8,
        "plugin_heads": 2,
        "x_encoder": "voronoi_star",
        "allow_future_x": True,
    }
    if backbone == "mtand":
        params.update(latent_dim=4, gen_hidden=8, te_dim=8, num_ref_points=8)
        cls = MTANQueryFusion
    else:
        params["n_layer"] = 1
        cls = TPatchQueryFusion if backbone == "tpatchgnn" else HiPatchQueryFusion
    m = cls(g, **params).eval()
    calibrate_voronoi_references(
        m, DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    )
    before = m(**batch)
    changed = sample.clone()
    changed[batch["batch_pred_mask"][0].bool(), 1] = float("nan")
    torch.testing.assert_close(before, m(**ds.collate_fn([(changed, origin)])))
    before.square().sum().backward()
    assert m.fusion.x_context.layers[0].density_gate.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters()
    )


@pytest.mark.parametrize("backbone", ["tpatchgnn", "hipatch", "mtand"])
def test_launcher_matched_pair_preserves_config_and_batch_override(tmp_path, backbone):
    import yaml

    data = tmp_path / "data"
    (data / "mimic/processed").mkdir(parents=True)
    output = tmp_path / "run"
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        TPATCH_DATA_ROOT=str(data),
        RUN_DIR=str(output),
        DRY_RUN="1",
        COMPARE="1",
        BATCH_SIZE="16",
    )
    run = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_voronoi_star_query_fusion.sh"),
            backbone,
            "mimic",
            "--device",
            "cpu",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    configs = list(output.glob("*/*/*/*/config.yaml"))
    assert len(configs) == 2
    assert {p.parent.name for p in configs} == {
        "voronoi_star",
        "voronoi_star_no_density",
    }
    for path in configs:
        config = yaml.safe_load(path.read_text())
        assert config["train_config"]["batch_size"] == 16
        assert config["model_config"]["model_params"]["x_encoder"] == "voronoi_star"
        assert config["model_config"]["model_params"].get(
            "use_density_encoding", True
        ) == ("no_density" not in path.parent.name)


def test_tiny_gaps_empty_X_and_batch_independence():
    x = example()
    m = core().eval()
    other = copy.deepcopy(x)
    other.times = 2 * x.times + 23
    batched = Observations(
        *(
            torch.cat([getattr(x, k), getattr(other, k)])
            for k in ["values", "times", "variables", "valid"]
        )
    )
    torch.testing.assert_close(m(batched)[:1], m(x))
    tiny = Observations(
        torch.ones((1, 3), dtype=torch.double),
        torch.tensor([[0.0, 1e-30, 1.0]], dtype=torch.double),
        torch.zeros((1, 3), dtype=torch.long),
        torch.ones((1, 3), dtype=torch.bool),
    )
    small = VoronoiStarTransformer(8, 1, heads=2).double().eval()
    small.fit_reference([tiny])
    assert torch.isfinite(small(tiny)).all()
    fusion = (
        DynamicLagFusion(
            6, 3, hidden=8, heads=2, x_encoder="voronoi_star", allow_future_x=True
        )
        .double()
        .eval()
    )
    fusion.x_context.fit_reference([x])
    x.valid.zero_()
    features = torch.randn(1, 2, 6, dtype=torch.double)
    times = torch.tensor([[1.0, 2.0]], dtype=torch.double)
    ids = torch.zeros((1, 2), dtype=torch.long)
    torch.testing.assert_close(
        fusion(features, times, ids, torch.ones_like(ids, dtype=torch.bool), x),
        features,
    )


@pytest.mark.parametrize(
    "backbone", ["TPatchQueryFusion", "HiPatchQueryFusion", "MTANQueryFusion"]
)
def test_one_epoch_cli_calibrates_and_saves_both_variants(tmp_path, backbone):
    import json

    from test_tpatch_records import make_data

    make_data(tmp_path, "activity")
    output = tmp_path / "run"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_query_fusion.py"),
            "--output",
            str(output),
            "--data-root",
            str(tmp_path),
            "--datasets",
            "activity",
            "--backbones",
            backbone,
            "--variants",
            "voronoi_star_no_density",
            "voronoi_star",
            "--small-model",
            "--max-samples-per-split",
            "1",
            "--batch-size",
            "1",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    summaries = json.loads((output / "summary.json").read_text())
    assert len(summaries) == 2 and all(r["exit_code"] == 0 for r in summaries)
    for path in output.glob("*/*/*/*/results/metrics.json"):
        d = json.loads(path.read_text())
        assert math.isfinite(d["test"]["mse"])
        assert d["protocol"]["fusion"]["density_reference_fitted_pairs"] > 0
        assert d["protocol"]["fusion"]["density_node_encoding"] is False
        assert len(d["training_history"]) == 1
