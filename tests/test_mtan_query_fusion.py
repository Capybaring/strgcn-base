"""Official decoder parity, data isolation, variational gradients, and CLI."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from test_tpatch_records import make_data

from data_processors import TPatchRecordProcessor
from datasets import MTANQueryDataset
from models.exogenous.mtan_adapter import MTANQueryFusion
from utils.scaler import TPatchInputScaler

ROOT = Path(__file__).resolve().parents[1]


def prepared(root, dataset="activity", inputs="FullExo"):
    g = {"device": "cpu"}
    data = TPatchRecordProcessor(
        g, dataset, data_root=root, max_samples_per_split=2
    ).process()
    scaler = TPatchInputScaler()
    scaler.fit(data["seen_data"])
    ds = MTANQueryDataset(
        scaler.transform(data["train_data"]),
        type="train",
        global_args=g,
        patch_size=dict(activity=300, mimic=8, physionet=8, ushcn=2)[dataset],
        use_exogenous=inputs != "NoExo",
        known_future_exogenous_var_ids=[] if inputs == "HistExo" else "all",
    )
    return g, ds


def model(g, mode="v1"):
    return MTANQueryFusion(
        g,
        hid_dim=8,
        latent_dim=4,
        gen_hidden=8,
        te_dim=8,
        num_ref_points=8,
        n_heads=1,
        fusion_mode=mode,
        plugin_hidden=8,
        plugin_heads=2,
    )


@pytest.mark.parametrize("dataset", ["activity", "mimic", "physionet", "ushcn"])
@pytest.mark.parametrize("mode", ["none", "mlp", "v1"])
def test_all_datasets_and_methods_backward_and_original_head_parity(
    tmp_path, dataset, mode
):
    make_data(tmp_path, dataset)
    g, ds = prepared(tmp_path, dataset)
    b = ds.collate_fn([ds[0], ds[1]])
    m = model(g, mode).eval()
    f = m.encode_queries(b)
    z = m.core.encode(
        b["batch_backbone_mtan_values"],
        b["batch_backbone_mtan_times"],
        b["batch_backbone_mtan_mask"],
        sample=False,
    )
    native = m.core.decoder(z, b["batch_backbone_queries"])[..., : len(m.target_ids)]
    adapted = m.decode(f)
    torch.testing.assert_close(adapted, native)
    # Gradients through the preserved output head match upstream decoding too.
    a = torch.autograd.grad(
        adapted.square().sum(), m.core.decoder.z0_to_obs[-1].weight, retain_graph=True
    )[0]
    e = torch.autograd.grad(
        native.square().sum(), m.core.decoder.z0_to_obs[-1].weight, retain_graph=True
    )[0]
    torch.testing.assert_close(a, e)
    m.train()
    pred = m(**b)
    q = b["batch_pred_mask"].bool()
    loss = (
        pred.squeeze(-1)[q] - b["batch_value"][q]
    ).square().mean() + m.auxiliary_loss()
    loss.backward()
    assert torch.isfinite(pred).all() and m.core.kl.item() >= 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.core.encoder.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.core.decoder.parameters()
    )
    if mode != "none":
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in m.fusion.parameters()
        )
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("inputs", ["FullExo", "HistExo", "NoExo"])
def test_future_labels_later_x_and_no_exogenous_identity(tmp_path, inputs):
    make_data(tmp_path, "activity")
    g, ds = prepared(tmp_path, inputs=inputs)
    sample, origin = ds[0]
    b = ds.collate_fn([(sample, origin)])
    m = model(g).eval()
    before = m(**b)
    altered = sample.clone()
    altered[b["batch_pred_mask"][0].bool(), 1] = 1e9
    changed = ds.collate_fn([(altered, origin)])
    for key in (
        "batch_backbone_mtan_values",
        "batch_backbone_mtan_times",
        "batch_backbone_mtan_mask",
    ):
        torch.testing.assert_close(b[key], changed[key])
    torch.testing.assert_close(before, m(**changed))
    later = torch.where((b["batch_node_type"][0] == 2) & b["batch_pad_mask"][0].bool())[
        0
    ]
    if len(later):
        ix = later[b["batch_timestamp"][0, later].argmax()]
        cut = b["batch_timestamp"][0, ix]
        altered = sample.clone()
        altered[ix, 1] += 100
        earlier = b["batch_pred_mask"].bool() & (b["batch_timestamp"] < cut)
        assert earlier.any()
        torch.testing.assert_close(
            before[earlier], m(**ds.collate_fn([(altered, origin)]))[earlier]
        )
    if inputs == "NoExo":
        f = m.encode_queries(b)
        torch.testing.assert_close(m.fuse(f, b), f)


def test_packing_duplicates_zeros_empty_and_padding(tmp_path):
    make_data(tmp_path, "activity")
    g, ds = prepared(tmp_path)
    sample, origin = ds[0]
    extra = sample[(sample[:, 2] == 0) & (sample[:, 3] == 0)][:1].clone()
    extra[:, 1] = 0
    augmented = torch.cat([sample, extra])
    b = ds.collate_fn([(augmented, origin), ds[1]])
    v, t, mask = (b["batch_backbone_mtan_" + k] for k in ("values", "times", "mask"))
    target = set(ds.packet["target_ids"])
    expected = sorted(
        (float(row[0]), int(row[2]), float(row[1]))
        for row in augmented
        if int(row[2]) in target and row[3] == 0
    )
    actual = sorted(
        (float(t[0, i]), ds.backbone_ids[j], float(v[0, i, j]))
        for i, j in torch.nonzero(mask[0], as_tuple=False).tolist()
    )
    assert actual == expected
    m = model(g).eval()
    a = m(**b)
    padded = copy.deepcopy(b)
    padded["batch_backbone_mtan_values"] = torch.cat(
        [v, torch.full_like(v[:, :1], float("nan"))], 1
    )
    padded["batch_backbone_mtan_times"] = torch.cat([t, torch.zeros_like(t[:, :1])], 1)
    padded["batch_backbone_mtan_mask"] = torch.cat(
        [mask, torch.zeros_like(mask[:, :1])], 1
    )
    torch.testing.assert_close(a, m(**padded))
    future = sample[sample[:, 3].bool()]
    empty = ds.collate_fn([(future, origin)])
    assert torch.isfinite(m(**empty)).all()


def test_latent_sampling_kl_formula_and_shared_backbone_initialization(tmp_path):
    make_data(tmp_path, "activity")
    g, ds = prepared(tmp_path)
    b = ds.collate_fn([ds[0]])
    torch.manual_seed(42)
    base = model(g, "none")
    torch.manual_seed(42)
    v1 = model(g, "v1")
    for key, value in base.core.state_dict().items():
        torch.testing.assert_close(value, v1.core.state_dict()[key])
    v1.eval()
    first = v1(**b)
    torch.testing.assert_close(first, v1(**b))
    enc = v1.core.encoder
    values = b["batch_backbone_mtan_values"]
    mask = b["batch_backbone_mtan_mask"]
    p = enc(torch.cat([values, mask], -1), b["batch_backbone_mtan_times"])
    mu, logvar = p.chunk(2, -1)
    expected = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    torch.testing.assert_close(v1.auxiliary_loss(), expected * v1.kl_weight)
    v1.train()
    a = v1.encode_queries(b)
    c = v1.encode_queries(b)
    assert not torch.equal(a, c)


def test_cli_twelve_runs_and_exports(tmp_path):
    for dataset in ("activity", "mimic", "physionet", "ushcn"):
        make_data(tmp_path / "data", dataset)
    output = tmp_path / "runs"
    cmd = [
        sys.executable,
        "scripts/run_query_fusion.py",
        "--data-root",
        str(tmp_path / "data"),
        "--output",
        str(output),
        "--backbones",
        "MTANQueryFusion",
        "--datasets",
        "activity",
        "mimic",
        "physionet",
        "ushcn",
        "--variants",
        "base",
        "mlp",
        "v1",
        "--seeds",
        "3407",
        "--epochs",
        "1",
        "--max-samples-per-split",
        "2",
        "--small-model",
        "--stream-logs",
    ]
    run = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert run.returncode == 0, run.stdout + run.stderr
    rows = json.loads((output / "summary.json").read_text())
    assert len(rows) == 12 and all(row["exit_code"] == 0 for row in rows)
    for path in output.glob("*/*/*/*/results"):
        metrics = json.loads((path / "metrics.json").read_text())
        assert metrics["protocol"]["fusion"]["kl_weight"] == 0.001
        for name in (
            "test_y_mask.npy",
            "test_var_idx.npy",
            "test_record_id.npy",
            "test_window_id.npy",
        ):
            assert (path / name).is_file()


@pytest.mark.parametrize("flag", ["--no-exogenous", "--history-only-exogenous"])
def test_main_input_ablation_flags(tmp_path, flag):
    make_data(tmp_path / "data", "activity")
    config = yaml.safe_load(
        (ROOT / "config/MTANQueryFusion/tpatch/activity.yaml").read_text()
    )
    config["data_config"]["data_processor_params"]["max_samples_per_split"] = 1
    config["model_config"]["model_params"].update(
        hid_dim=8,
        latent_dim=4,
        gen_hidden=8,
        te_dim=8,
        num_ref_points=8,
        plugin_hidden=8,
        plugin_heads=2,
    )
    config["train_config"]["trainer_params"]["max_epoch_num"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    run = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config_path",
            str(path),
            "--tpatch-data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            flag,
            "--model_save_dir_path",
            str(tmp_path / "model"),
            "--result_save_dir_path",
            str(tmp_path / "result"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((tmp_path / "result/metrics.json").read_text())
    assert report["protocol"]["known_future_exogenous_var_ids"] == []
    assert report["protocol"]["use_exogenous"] == (flag != "--no-exogenous")
