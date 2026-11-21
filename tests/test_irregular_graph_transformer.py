"""Regular-zero proof cases, geometry semantics and full adapter information flow."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from test_query_fusion import prepared, toy
from test_tpatch_records import make_data

from models.exogenous.fusion import DynamicLagFusion, Observations
from models.exogenous.irregular_graph_transformer import (
    CenteredGapEncoding,
    sampling_geometry,
)

ROOT = Path(__file__).resolve().parents[1]


def geometry(t, ids=None):
    if ids is None:
        ids = torch.zeros_like(t, dtype=torch.long)
    return sampling_geometry(t, ids, torch.ones_like(t, dtype=torch.bool))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_regular_zero_with_boundaries_duplicates_and_independent_variable_rates(dtype):
    t = torch.tensor([[0.0, 0.1, 0.2, 0.2, 0.3, 0.5, 0.9, 1.3, 0.7]], dtype=dtype)
    ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 2]])
    z, edges, density = geometry(t, ids)
    assert torch.count_nonzero(z) == torch.count_nonzero(density) == 0
    torch.testing.assert_close(edges[0, 0], t.new_tensor([0, 1]))
    torch.testing.assert_close(edges[0, 4], t.new_tensor([1, 0]))
    assert not edges[0, -1].any()
    encoder = CenteredGapEncoding(8).to(dtype)
    with torch.no_grad():
        for p in encoder.parameters():
            p.fill_(0.7)  # Nonzero biases must cancel, at boundaries too.
    assert torch.count_nonzero(encoder(z, edges)) == 0
    encoder(z, edges).sum().backward()
    assert all(torch.count_nonzero(p.grad) == 0 for p in encoder.parameters())


def test_direction_density_scale_shift_order_duplicates_and_batch_independence():
    t = torch.tensor([[0.0, 1.0, 4.0, 6.0, 6.0]], dtype=torch.float64)
    z, edges, d = geometry(t)
    # Mean unique-time gap=2: a short left gap and long right gap at t=1.
    torch.testing.assert_close(z[0, 1], t.new_tensor([-0.5, 0.5]))
    torch.testing.assert_close(d[0, 1], t.new_tensor([2.0, 2 / 3]).log())
    assert d[0, 1, 0] > 0 and d[0, 1, 1] < 0
    assert d[0, 0, 0] == d[0, -1, 1] == 0
    for a, b in zip((z, edges, d), geometry(3 * t + 17)):
        torch.testing.assert_close(a, b)
    # Reversing physical time swaps left/right geometry.
    for a, b in zip((z, edges, d), geometry(6 - t)):
        torch.testing.assert_close(a.flip(-1), b)
    order = torch.tensor([3, 1, 0, 4, 2])
    for a, b in zip((z, edges, d), geometry(t[:, order])):
        torch.testing.assert_close(a[:, order], b)
    for a in (z, edges, d):
        torch.testing.assert_close(a[:, -1], a[:, -2])
    for a, b in zip((z, edges, d), geometry(torch.cat([t, 7 * t + 50]))):
        torch.testing.assert_close(a, b[:1])
    encoder = CenteredGapEncoding(8).double()
    assert encoder(z, edges).abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_regular_float_roundoff_and_visible_small_irregularity(dtype):
    t = torch.linspace(0.1, 0.9, 100, dtype=dtype)[None]
    assert all(torch.count_nonzero(a) == 0 for a in (geometry(t)[0], geometry(t)[2]))
    t[:, 50] += 0.0001
    assert geometry(t)[0].abs().sum() > 0


def fusion(**kwargs):
    return DynamicLagFusion(
        6, 3, hidden=8, heads=2, x_encoder="irgraph", allow_future_x=True, **kwargs
    )


def inputs(dtype=torch.double):
    x, _ = toy()
    x.values, x.times = x.values.to(dtype), x.times.to(dtype)
    f = torch.randn(1, 2, 6, dtype=dtype)
    t = torch.tensor([[0.01, 0.05]], dtype=dtype)  # All X are in the query future.
    ids = torch.ones(1, 2, dtype=torch.long)
    return x, f, t, ids, ids.bool()


def test_same_weight_regular_graph_degeneracy_including_xy_density():
    m = fusion().double().eval()
    x, f, t, ids, valid = inputs()
    x.times = torch.tensor([[0.1, 0.3, 0.5]], dtype=torch.double)
    plain = copy.deepcopy(m)
    plain.x_context.use_gap_encoding = plain.x_context.use_density_encoding = False
    torch.testing.assert_close(
        m(f, t, ids, valid, x), plain(f, t, ids, valid, x), atol=0, rtol=0
    )
    assert torch.count_nonzero(m.x_context.cross_density_key(geometry(x.times)[2])) == 0


def test_future_x_geometry_gradients_chunk_checkpoint_and_empty_padding_identity():
    m = fusion(chunk_size=1).double().train()
    x, f, t, ids, valid = inputs()
    x.values.requires_grad_()
    actual = m(f, t, ids, valid, x)
    params = [x.values] + list(m.x_context.parameters())
    ga = torch.autograd.grad(actual.square().sum(), params, allow_unused=True)
    assert (ga[0].abs() > 1e-10).all()  # Future X is accessible, not masked.
    for name, p in m.x_context.named_parameters():
        if name in (
            "gap_encoding.net.0.weight",
            "layers.0.q.weight",
            "layers.0.density_q.weight",
            "layers.0.density_k.weight",
            "cross_density_key.weight",
        ):
            g = ga[1 + list(dict(m.x_context.named_parameters())).index(name)]
            assert g is not None and g.abs().sum() > 1e-12, name
    m.chunk_size = m.x_context.layers[0].chunk_size = 100
    expected = m(f, t, ids, valid, x)
    ge = torch.autograd.grad(expected.square().sum(), params, allow_unused=True)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(ga, ge):
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)
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
    assert torch.isfinite(m.x_context(m._clean(padded))).all()


def test_node_and_density_ablations_both_change_irregular_output():
    m = fusion().double().eval()
    x, f, t, ids, valid = inputs()
    full = m(f, t, ids, valid, x)
    for flag in ("use_gap_encoding", "use_density_encoding"):
        ablation = copy.deepcopy(m)
        setattr(ablation.x_context, flag, False)
        assert not torch.allclose(
            full, ablation(f, t, ids, valid, x), atol=1e-10, rtol=1e-10
        )


@pytest.mark.parametrize("backbone", ["tpatchgnn", "hipatch", "mtand"])
@pytest.mark.parametrize("dataset", ["activity", "mimic", "physionet", "ushcn"])
def test_backbones_keep_encoder_decoder_labels_and_gradients(
    tmp_path, backbone, dataset
):
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
        x_encoder="irgraph",
        allow_future_x=True,
    )
    if backbone == "mtand":
        params.update(latent_dim=4, gen_hidden=8, te_dim=8, num_ref_points=8)
        cls = MTANQueryFusion
    else:
        params["n_layer"] = 1
        cls = TPatchQueryFusion if backbone == "tpatchgnn" else HiPatchQueryFusion
    m = cls(g, **params).eval()
    features = m.encode_queries(b)
    if backbone != "mtand":
        native = m.core.forecasting(
            b["batch_backbone_queries"],
            b["batch_backbone_values"],
            b["batch_backbone_times"],
            b["batch_backbone_mask"],
        )[0]
        torch.testing.assert_close(m.decode(features), native[..., : len(m.target_ids)])
    before = m(**b)
    altered = sample.clone()
    altered[b["batch_pred_mask"][0].bool(), 1] = float("nan")
    torch.testing.assert_close(before, m(**ds.collate_fn([(altered, origin)])))
    before.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters()
    )
    assert m.fusion.x_context.layers[0].q.weight.grad.abs().sum() > 0
    assert m.fusion.x_context.layers[0].u.weight.grad.abs().sum() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    assert g["query_fusion_protocol"]["x_visibility"] == "all supplied observed X"


@pytest.mark.parametrize(
    "backbone", ["TPatchQueryFusion", "HiPatchQueryFusion", "MTANQueryFusion"]
)
@pytest.mark.parametrize(
    "method", ["graph", "irgraph", "irgraph_no_node", "irgraph_no_density"]
)
def test_dist_new_methods(tmp_path, backbone, method):
    from test_dist_query_fusion import test_dist_methods_backward_and_label_isolation

    test_dist_methods_backward_and_label_isolation(tmp_path, backbone, method)


@pytest.mark.parametrize("compare", [False, True])
def test_shell_matrix_cuda_batch_defaults_and_no_overwrite(tmp_path, compare):
    data = tmp_path / "data with spaces"
    for ds in ("activity", "mimic", "physionet", "ushcn"):
        (data / ds / "processed").mkdir(parents=True)
    dist = tmp_path / "DIST.npz"
    dist.touch()
    output = tmp_path / "results"
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        TPATCH_DATA_ROOT=str(data),
        DIST_DATA_PATH=str(dist),
        RUN_DIR=str(output),
        DRY_RUN="1",
        COMPARE="1" if compare else "0",
    )
    for key in ("GPU", "DEVICE", "CUDA_VISIBLE_DEVICES", "BATCH_SIZE"):
        env.pop(key, None)
    if compare:
        env["BATCH_SIZE"] = "2"
    cmd = [
        "bash",
        str(ROOT / "scripts/run_irgraph_query_fusion.sh"),
        "--device",
        "cuda:1",
    ]
    run = subprocess.run(
        cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr
    configs = list(output.glob("*/*/*/*/config.yaml"))
    assert len(configs) == (105 if compare else 15)
    for path in configs:
        c = yaml.safe_load(path.read_text())
        dataset = path.parts[-4]
        config_path = (
            ROOT
            / "config"
            / path.parts[-5]
            / ("processed/DIST.yaml" if dataset == "dist" else f"tpatch/{dataset}.yaml")
        )
        original = yaml.safe_load(config_path.read_text())
        assert c["train_config"]["batch_size"] == (
            2 if compare else original["train_config"]["batch_size"]
        )
        if path.parent.name in (
            "graph",
            "irgraph",
            "irgraph_no_node",
            "irgraph_no_density",
        ):
            assert c["model_config"]["model_params"]["allow_future_x"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["arguments"]["device"] == "cuda:1"
    assert manifest["cuda_visible_devices"] is None
    again = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    assert again.returncode != 0 and "FileExistsError" in again.stderr


@pytest.mark.parametrize(
    "method", ["graph", "irgraph", "irgraph_no_node", "irgraph_no_density"]
)
def test_single_script_new_method(tmp_path, method):
    from test_device_selection import launch

    run, output = launch(tmp_path, ["hipatch", "mimic", method, "--device", "cuda:1"])
    assert run.returncode == 0, run.stdout + run.stderr
    assert len(list(output.glob("*/*/*/*/config.yaml"))) == 1


def test_reject_hidden_future_visibility_and_unsupported_transfer():
    with pytest.raises(ValueError, match="allow_future_x"):
        DynamicLagFusion(6, 3, x_encoder="irgraph")
    with pytest.raises(ValueError, match="V2"):
        fusion(mode="v2")
