"""Real official operators, pre-head parity, label isolation and launch matrices."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

import models
from scripts.run_query_fusion import VARIANTS
from test_mtan_query_fusion import prepared
from test_tpatch_records import make_data

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["GraFITiQueryFusion", "CRUQueryFusion", "LatentODEQueryFusion"]
METHODS = ["base", "voronoi_h_no_density", "voronoi_h"]


def small_params(method):
    return dict(
        hid_dim=8,
        latent_dim=4,
        n_heads=2,
        n_layer=1,
        ode_hidden=8,
        gru_units=8,
        num_basis=2,
        bandwidth=1,
        plugin_hidden=8,
        plugin_heads=2,
        **VARIANTS[method],
    )


@pytest.mark.parametrize("backbone", MODELS)
@pytest.mark.parametrize("dataset", ["activity", "mimic", "physionet", "ushcn"])
@pytest.mark.parametrize("method", METHODS)
def test_adapters_native_head_label_isolation_gradients(
    tmp_path, backbone, dataset, method
):
    torch.manual_seed(3407)
    make_data(tmp_path, dataset)
    g, ds = prepared(tmp_path, dataset)
    sample, origin = ds[0]
    b = ds.collate_fn([(sample, origin)])
    m = getattr(models, backbone)(g, **small_params(method)).eval()
    features = m.encode_queries(b)
    native = m.decode(features)
    if backbone == "GraFITiQueryFusion":
        torch.testing.assert_close(native, m.core.model.output(features).squeeze(-1))
    else:
        n = len(m.target_ids)
        for j in range(n):
            f = features[:, :, j].to(m.core.output_layer.weight.dtype)
            if backbone == "CRUQueryFusion":
                for layer in m.core.decoder._hidden_layers_mean:
                    f = layer(f)
            torch.testing.assert_close(native[:, :, j], m.core.output_layer(f)[:, :, j])
    pred = m(**b)
    changed = sample.clone()
    changed[b["batch_pred_mask"][0].bool(), 1] = float("nan")
    torch.testing.assert_close(pred, m(**ds.collate_fn([(changed, origin)])))
    loss = m.forecast_loss(pred.squeeze(-1), b) + m.auxiliary_loss()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters()
    )
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    if method != "base":
        # An upstream ReLU decoder can have zero input derivative at a random
        # initialization. Test the plugin's gradient path independently as well.
        gradient = torch.autograd.grad(
            m.fuse(features, b).square().sum(), m.fusion.out.weight
        )[0]
        assert gradient.abs().sum() > 0
        assert not m.fusion.x_context.layers[0].density_relation
    if backbone == "CRUQueryFusion":
        assert (m._forecast_variance > 0).all()
        assert m.core.decoder._out_layer_var.weight.grad.abs().sum() > 0
    if backbone == "LatentODEQueryFusion":
        assert m._kl >= 0 and m.core.encoder.transform_z0[0].weight.grad is not None


@pytest.mark.parametrize("backbone", MODELS)
def test_batch_padding_and_permutation_do_not_change_sample_predictions(
    tmp_path, backbone
):
    make_data(tmp_path, "mimic")
    g, ds = prepared(tmp_path, "mimic")
    m = getattr(models, backbone)(g, **small_params("voronoi_h")).eval()
    single = ds.collate_fn([ds[0]])
    batch = ds.collate_fn([ds[0], ds[1]])
    p = m(**single)[0]
    actual = m(**batch)[0, : len(p)]
    torch.testing.assert_close(p, actual, atol=2e-6, rtol=2e-5)
    # Without X, shared fusion must leave query representations untouched.
    f = m.encode_queries(batch)
    no_x = dict(batch)
    no_x["batch_node_type"] = torch.ones_like(batch["batch_node_type"])
    torch.testing.assert_close(m.fuse(f, no_x), f)


@pytest.mark.parametrize("backbone", MODELS)
@pytest.mark.parametrize("method", METHODS)
def test_dist_native_adapters(tmp_path, backbone, method):
    from test_dist_query_fusion import test_dist_methods_backward_and_label_isolation

    test_dist_methods_backward_and_label_isolation(tmp_path, backbone, method)


@pytest.mark.parametrize(
    "name,model", list(zip(["grafiti", "cru", "latentode"], MODELS))
)
@pytest.mark.parametrize("compare", [False, True])
def test_batch_launcher_config_seed_batch_device_and_no_overwrite(
    tmp_path, name, model, compare
):
    data = tmp_path / "data with spaces"
    for ds in ["activity", "mimic", "physionet", "ushcn"]:
        (data / ds / "processed").mkdir(parents=True)
    dist = tmp_path / "DIST.npz"
    dist.touch()
    output = tmp_path / "output"
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        TPATCH_DATA_ROOT=str(data),
        DIST_DATA_PATH=str(dist),
        RUN_DIR=str(output),
        DRY_RUN="1",
        COMPARE=str(int(compare)),
    )
    for key in ["GPU", "DEVICE", "CUDA_VISIBLE_DEVICES", "BATCH_SIZE"]:
        env.pop(key, None)
    if compare:
        env["BATCH_SIZE"] = "2"
    cmd = [
        "bash",
        str(ROOT / "scripts/run_irregular_voronoi_h.sh"),
        name,
        "--device",
        "cuda:1",
    ]
    run = subprocess.run(
        cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr
    configs = list(output.glob("*/*/*/*/config.yaml"))
    assert len(configs) == (15 if compare else 5)
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        dataset = path.parts[-4]
        source = (
            ROOT
            / "config"
            / model
            / ("processed/DIST.yaml" if dataset == "dist" else f"tpatch/{dataset}.yaml")
        )
        original = yaml.safe_load(source.read_text())
        assert cfg["train_config"]["batch_size"] == (
            2 if compare else original["train_config"]["batch_size"]
        )
        assert cfg["train_config"]["random_seed"] == 3407
        assert cfg["model_config"]["model_name"] == model
    manifest = json.loads((output / "manifest.json").read_text())
    assert (
        manifest["arguments"]["device"] == "cuda:1"
        and manifest["cuda_visible_devices"] is None
    )
    again = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    assert again.returncode != 0 and "FileExistsError" in again.stderr


def test_grafiti_official_forward_equals_exposed_query_head():
    from baseline.irregular_backbones.cores import GraFITiCore

    core = GraFITiCore(2, 8, 1, 2).eval()
    t = torch.tensor([0.1, 0.4])
    q = torch.tensor([0.7, 0.9])
    values = torch.tensor([[1.0, 0.0], [2.0, 3.0]])
    mask = values != 0
    f = core.query_features(values, t, mask, q)
    times = torch.cat([t, q])[None]
    vals = torch.cat([values, torch.zeros(2, 2)])[None]
    obs = torch.cat([mask, torch.zeros_like(mask)])[None].float()
    target = torch.cat([torch.zeros_like(mask), torch.ones_like(mask)])[None].float()
    original = core.model(times, vals, obs, target)[0, 2:]
    torch.testing.assert_close(
        core.model.output(f).squeeze(-1), original, atol=0, rtol=0
    )


def test_latentode_initial_state_anchored_to_history_not_query():
    from baseline.irregular_backbones.cores import LatentODECore

    core = LatentODECore(1, 8, 4, 8, gru_units=8).eval()

    # Exact constant derivative makes an incorrect first-query anchor observable.
    class Encoder(torch.nn.Module):
        def forward(self, data, times):
            return data.new_zeros(1, 1, 4), data.new_ones(1, 1, 4)

    class Gradient(torch.nn.Module):
        def forward(self, t, y):
            return torch.ones_like(y)

    core.encoder = Encoder()
    core.solver.ode_func = Gradient()
    f = core.query_features(
        torch.ones(1, 1),
        torch.tensor([0.2]),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([0.6, 0.9]),
    )
    torch.testing.assert_close(
        f, torch.tensor([[0.4] * 4, [0.7] * 4]), atol=2e-5, rtol=2e-5
    )


def test_cru_query_states_and_complete_decoder_match_native_filter():
    from baseline.irregular_backbones.cores import CRUCore
    from baseline.irregular_backbones.vendor.cru.CRUCell import var_activation

    core = CRUCore(2, 8, 4, 2, 1).eval()
    values = torch.tensor([[1.0, 0.0], [2.0, 3.0]])
    t = torch.tensor([0.1, 0.4])
    q = torch.tensor([0.7, 0.9])
    mask = values != 0
    states = core.query_features(values, t, mask, q)
    variance = core.forecast_variance
    v = torch.cat([values.double(), torch.zeros(2, 2, dtype=torch.double)])
    y, yvar = core.encoder(v)
    post, cov, *_ = core.layer(
        y[None],
        yvar[None],
        core.initial_mean,
        [
            var_activation(core.initial_log_upper),
            var_activation(core.initial_log_lower),
            core.initial_side,
        ],
        obs_valid=torch.tensor([[True, True, False, False]]),
        time_points=torch.cat([t, q])[None].double(),
    )
    torch.testing.assert_close(states, post[0, 2:], atol=0, rtol=0)
    expected_mean, expected_var = core.decoder(post[0, 2:], torch.cat(cov, -1)[0, 2:])
    actual = states
    for layer in core.decoder._hidden_layers_mean:
        actual = layer(actual)
    torch.testing.assert_close(core.output_layer(actual), expected_mean, atol=0, rtol=0)
    torch.testing.assert_close(variance, expected_var, atol=0, rtol=0)


@pytest.mark.parametrize("backbone", MODELS)
def test_future_y_has_zero_input_gradient_and_future_x_reaches_queries(
    tmp_path, backbone
):
    torch.manual_seed(3407)
    make_data(tmp_path, "activity")
    g, ds = prepared(tmp_path, "activity")
    b = ds.collate_fn([ds[0]])
    b["batch_value"].requires_grad_()
    m = getattr(models, backbone)(g, **small_params("voronoi_h")).eval()
    if backbone == "CRUQueryFusion":
        # Isolate information flow from incidental dead ReLU activations.
        with torch.no_grad():
            for layer in m.core.decoder._hidden_layers_mean:
                if isinstance(layer, torch.nn.Linear):
                    layer.bias.fill_(5.0)
    pred = m(**b)
    gradient = torch.autograd.grad(pred.square().sum(), b["batch_value"])[0]
    future_y = b["batch_pred_mask"].bool()
    assert torch.count_nonzero(gradient[future_y]) == 0
    future_x = (b["batch_node_type"] == 2) & b["batch_pad_mask"].bool()
    assert future_x.any() and gradient[future_x].abs().sum() > 0
