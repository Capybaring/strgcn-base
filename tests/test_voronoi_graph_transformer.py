"""Density formulas, pairwise bias, adapter isolation and batch launch plans."""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from models.exogenous.fusion import DynamicLagFusion, Observations
from models.exogenous.irregular_graph_transformer import voronoi_density_bias
from models.exogenous.voronoi_graph_transformer import voronoi_log_density
from test_irregular_graph_transformer import inputs
from test_tpatch_records import make_data

ROOT = Path(__file__).resolve().parents[1]


def geometry(t, ids=None, valid=None):
    if ids is None:
        ids = torch.zeros_like(t, dtype=torch.long)
    if valid is None:
        valid = torch.ones_like(t, dtype=torch.bool)
    return voronoi_log_density(t, ids, valid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_regular_zero_boundaries_variable_rates_duplicates_singleton(dtype):
    t = torch.tensor([[0, 0.1, 0.2, 0.2, 0.3, 0.5, 0.9, 1.3, 0.7]], dtype=dtype)
    ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 2]])
    s = geometry(t, ids)
    assert torch.count_nonzero(s) == 0
    assert (
        torch.count_nonzero(voronoi_density_bias(s, s, t.new_tensor([-2.0, 1.0]))) == 0
    )
    t = torch.linspace(0.1, 0.9, 100, dtype=dtype)[None]
    assert torch.count_nonzero(geometry(t)) == 0
    t[:, 50] += 0.0001
    assert torch.count_nonzero(geometry(t)) > 0


def test_voronoi_formula_compensation_and_single_sided_endpoints():
    t = torch.tensor([[0.0, 1.0, 4.0, 6.0, 6.0]], dtype=torch.double)
    # mu=2; rho=[1, 1/2, 2/5, 1/2, 1/2]. Unequal (1,3) has baseline density.
    expected = t.new_tensor([[2.0, 1.0, 0.8, 1.0, 1.0]]).log()
    torch.testing.assert_close(geometry(t), expected)
    torch.testing.assert_close(geometry(3 * t + 17), expected)
    torch.testing.assert_close(geometry(6 - t), expected)
    order = torch.tensor([3, 1, 0, 4, 2])
    torch.testing.assert_close(geometry(t[:, order]), expected[:, order])
    torch.testing.assert_close(
        geometry(torch.cat([t, 7 * t + 50])), expected.expand(2, -1)
    )
    # The exact local baseline is also possible inside a globally irregular grid.
    t = torch.tensor([[0.0, 0.5, 1.5, 2.5, 4.0]], dtype=torch.double)
    assert geometry(t)[0, 2] == 0


def test_density_bias_matches_log_gaussian_and_keeps_regular_irregular_relations():
    s = torch.tensor([[0.0, math.log(2), -math.log(2)]], dtype=torch.double)
    raw = torch.tensor(
        [math.log(math.expm1(1.0)), -2.0], dtype=torch.double, requires_grad=True
    )
    bias = voronoi_density_bias(s, s, raw)
    assert (bias <= 0).all()
    torch.testing.assert_close(bias, bias.transpose(-1, -2))
    assert torch.count_nonzero(bias.diagonal(dim1=-2, dim2=-1)) == 0
    assert bias[0, 0, 0, 1] < 0  # A zero endpoint must not annul the relation.
    torch.testing.assert_close(bias[0, 0, 0, 1], -s[0, 1].square())
    torch.testing.assert_close(bias, voronoi_density_bias(s + 3, s + 3, raw))
    bias.sum().backward()
    assert torch.isfinite(raw.grad).all() and (raw.grad != 0).all()


def test_near_coincident_density_is_finite_unclipped_and_padding_is_ignored():
    t = torch.tensor([[0.0, 1e-30, 1.0, float("nan")]], dtype=torch.float32)
    valid = torch.tensor([[True, True, True, False]])
    s = geometry(t, valid=valid)
    assert torch.isfinite(s).all() and s[0, 0] > 8
    assert s[0, -1] == 0


def fusion(**kwargs):
    return DynamicLagFusion(
        6, 3, hidden=8, heads=2, x_encoder="voronoi", allow_future_x=True, **kwargs
    )


@pytest.mark.parametrize("location", ["attention", "node"])
def test_regular_exact_ablation_equivalence_no_legacy_branches(location):
    m = fusion(density_location=location).double().eval()
    x, f, t, ids, valid = inputs()
    x.times = torch.tensor([[0.1, 0.3, 0.5]], dtype=torch.double)
    plain = copy.deepcopy(m)
    plain.x_context.use_density_encoding = False
    torch.testing.assert_close(
        m(f, t, ids, valid, x), plain(f, t, ids, valid, x), atol=0, rtol=0
    )
    names = dict(m.x_context.named_parameters())
    assert not any(
        k in name
        for name in names
        for k in ("gap_encoding", "density_q", "density_k", "cross_density_key")
    )
    protocol = m.protocol()
    assert not protocol["gap_node_encoding"]
    if location == "attention":
        assert "independent X attention logit only" in protocol["density_injection"]
    else:
        assert protocol["density_node_encoding"]
        assert not protocol["density_attention_encoding"]
        assert protocol["density_bias"] == "none"
        assert not any("density_log_scale" in name for name in names)
        assert m.x_context.node_fusion[0].in_features == 3 * m.hidden + 1


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("location", ["attention", "node"])
def test_future_x_gradients_chunk_checkpoint_padding_permutation_and_empty_identity(
    layers,
    location,
):
    m = (
        fusion(chunk_size=1, x_layers=layers, density_location=location)
        .double()
        .train()
    )
    x, f, t, ids, valid = inputs()
    x.values.requires_grad_()
    actual = m(f, t, ids, valid, x)
    params = [x.values] + list(m.x_context.parameters())
    ga = torch.autograd.grad(actual.square().sum(), params, allow_unused=True)
    assert (ga[0].abs() > 1e-10).all()
    name_list = list(dict(m.x_context.named_parameters()))
    density_parameter = (
        "layers.0.density_log_scale"
        if location == "attention"
        else "node_fusion.0.weight"
    )
    for name in ("layers.0.q.weight", density_parameter):
        grad = ga[1 + name_list.index(name)]
        assert grad is not None and grad.abs().sum() > 1e-12
    if location == "node":
        # Last input column is the analytic scalar density, not a learned estimator.
        assert ga[1 + name_list.index(density_parameter)][:, -1].abs().sum() > 1e-12
    m.chunk_size = 100
    for layer in m.x_context.layers:
        layer.chunk_size = 100
    expected = m(f, t, ids, valid, x)
    ge = torch.autograd.grad(expected.square().sum(), params, allow_unused=True)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(ga, ge):
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)
    plain = copy.deepcopy(m)
    plain.x_context.use_density_encoding = False
    assert not torch.allclose(
        actual, plain(f, t, ids, valid, x), atol=1e-10, rtol=1e-10
    )
    padded = Observations(
        torch.cat([x.values, x.values.new_tensor([[float("nan")]])], 1),
        torch.cat([x.times, x.times.new_tensor([[float("nan")]])], 1),
        torch.tensor([[0, 0, 0, 999]]),
        torch.tensor([[1, 1, 1, 0]]).bool(),
    )
    torch.testing.assert_close(actual, m(f, t, ids, valid, padded))
    order = torch.tensor([2, 0, 1])
    permuted = Observations(
        *(getattr(x, k)[:, order] for k in ("values", "times", "variables", "valid"))
    )
    torch.testing.assert_close(actual, m(f, t, ids, valid, permuted))
    padded.valid[:] = False
    torch.testing.assert_close(f, m(f, t, ids, valid, padded))


@pytest.mark.parametrize("backbone", ["tpatchgnn", "hipatch", "mtand"])
@pytest.mark.parametrize("dataset", ["activity", "mimic", "physionet", "ushcn"])
@pytest.mark.parametrize("location", ["attention", "node"])
def test_adapter_original_head_future_label_isolation_and_backward(
    tmp_path, backbone, dataset, location
):
    from test_query_fusion import prepared
    from test_mtan_query_fusion import prepared as mtan_prepared
    from models.exogenous.adapter import HiPatchQueryFusion, TPatchQueryFusion
    from models.exogenous.mtan_adapter import MTANQueryFusion

    make_data(tmp_path, dataset)
    g, ds = (mtan_prepared if backbone == "mtand" else prepared)(tmp_path, dataset)
    sample, origin = ds[0]
    b = ds.collate_fn([(sample, origin)])
    params = dict(
        hid_dim=8,
        n_heads=1,
        plugin_hidden=8,
        plugin_heads=2,
        fusion_mode="v1",
        x_encoder="voronoi",
        allow_future_x=True,
        density_location=location,
    )
    if backbone == "mtand":
        params.update(latent_dim=4, gen_hidden=8, te_dim=8, num_ref_points=8)
        cls = MTANQueryFusion
    else:
        params["n_layer"] = 1
        cls = TPatchQueryFusion if backbone == "tpatchgnn" else HiPatchQueryFusion
    m = cls(g, **params).eval()
    if backbone != "mtand":
        native = m.core.forecasting(
            b["batch_backbone_queries"],
            b["batch_backbone_values"],
            b["batch_backbone_times"],
            b["batch_backbone_mask"],
        )[0]
        torch.testing.assert_close(
            m.decode(m.encode_queries(b)), native[..., : len(m.target_ids)]
        )
    before = m(**b)
    changed = sample.clone()
    changed[b["batch_pred_mask"][0].bool(), 1] = float("nan")
    torch.testing.assert_close(before, m(**ds.collate_fn([(changed, origin)])))
    before.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters()
    )
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    assert g["query_fusion_protocol"]["x_visibility"] == "all supplied observed X"


@pytest.mark.parametrize(
    "backbone", ["TPatchQueryFusion", "HiPatchQueryFusion", "MTANQueryFusion"]
)
@pytest.mark.parametrize(
    "method", ["voronoi", "voronoi_no_density", "voronoi_h", "voronoi_h_no_density"]
)
def test_dist_adapter(tmp_path, backbone, method):
    from test_dist_query_fusion import test_dist_methods_backward_and_label_isolation

    test_dist_methods_backward_and_label_isolation(tmp_path, backbone, method)


@pytest.mark.parametrize("backbone", ["tpatchgnn", "hipatch", "mtand", None])
@pytest.mark.parametrize("compare", [False, True])
@pytest.mark.parametrize("variant", ["voronoi", "voronoi_h"])
def test_three_batches_cuda_default_configs_override_and_no_overwrite(
    tmp_path, backbone, compare, variant
):
    data = tmp_path / "data with spaces"
    for dataset in ("activity", "mimic", "physionet", "ushcn"):
        (data / dataset / "processed").mkdir(parents=True)
    dist = tmp_path / "DIST.npz"
    dist.touch()
    output = tmp_path / "runs"
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        TPATCH_DATA_ROOT=str(data),
        DIST_DATA_PATH=str(dist),
        RUN_DIR=str(output),
        DRY_RUN="1",
        COMPARE=str(int(compare)),
    )
    for key in (
        "GPU",
        "DEVICE",
        "CUDA_VISIBLE_DEVICES",
        "BATCH_SIZE",
        "VORONOI_VARIANT",
    ):
        env.pop(key, None)
    if compare:
        env["BATCH_SIZE"] = "2"
    script = (
        "run_voronoi_h_query_fusion.sh"
        if variant == "voronoi_h"
        else "run_voronoi_query_fusion.sh"
    )
    cmd = ["bash", str(ROOT / "scripts" / script)]
    if backbone is not None:
        cmd.append(backbone)
    cmd += ["--device", "cuda:1"]
    run = subprocess.run(
        cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr
    configs = list(output.glob("*/*/*/*/config.yaml"))
    assert len(configs) == (10 if compare else 5)
    model = dict(
        tpatchgnn="TPatchQueryFusion",
        hipatch="HiPatchQueryFusion",
        mtand="MTANQueryFusion",
    )[backbone or "tpatchgnn"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["arguments"]["device"] == "cuda:1"
    assert manifest["cuda_visible_devices"] is None
    for path in configs:
        c = yaml.safe_load(path.read_text())
        dataset = path.parts[-4]
        source = (
            ROOT
            / "config"
            / model
            / ("processed/DIST.yaml" if dataset == "dist" else f"tpatch/{dataset}.yaml")
        )
        original = yaml.safe_load(source.read_text())
        assert path.parts[-5] == model
        assert c["train_config"]["random_seed"] == 3407
        assert c["train_config"]["batch_size"] == (
            2 if compare else original["train_config"]["batch_size"]
        )
        assert c["model_config"]["model_params"]["x_encoder"] == "voronoi"
        assert path.parent.name in (
            (variant, f"{variant}_no_density") if compare else (variant,)
        )
        if variant == "voronoi_h":
            assert c["model_config"]["model_params"]["density_location"] == "node"
    again = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    assert again.returncode != 0 and "FileExistsError" in again.stderr
