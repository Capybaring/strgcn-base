"""DIST protocols, all 15 query-fusion combinations and CLI exports."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

import datasets
import models
from data_processors import ExEnProcessedProcessor
from utils.scaler import ExEnProcessedMinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ("TPatchQueryFusion", "HiPatchQueryFusion", "MTANQueryFusion")
METHODS = ("original", "base", "mlp", "v1", "strgcn")


def archive(path):
    config = yaml.safe_load(
        (ROOT / "config/HiPatchQueryFusion/processed/DIST.yaml").read_text()
    )
    names = config["data_config"]["data_processor_params"]["target_names"]
    names = names + [n.replace(".PV", ".OP") for n in names]
    t = np.arange(0, 90001, 300, dtype=float)
    x = (
        2 + np.sin(t[:, None] / 700 + np.arange(18)[None, :]) + t[:, None] / 90000
    ).astype("float32")
    mask = np.ones_like(x, dtype=bool)
    x[0, 0] = 0
    mask[1, 1] = False
    np.savez(path, time=t, values=x, mask=mask, names=names)
    return config


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("method", METHODS)
def test_dist_methods_backward_and_label_isolation(tmp_path, backbone, method):
    from scripts.run_query_fusion import VARIANTS

    path = tmp_path / "DIST.npz"
    archive(path)
    c = yaml.safe_load((ROOT / f"config/{backbone}/processed/DIST.yaml").read_text())
    g = dict(c["global_config"], device="cpu", exen_data_path=str(path))
    data = ExEnProcessedProcessor(
        g, **c["data_config"]["data_processor_params"]
    ).process()
    scaler = ExEnProcessedMinMaxScaler()
    scaler.fit(data["train_data"])
    dp = dict(c["data_config"]["dataset_params"])
    name = c["data_config"]["dataset_name"]
    if method == "original":
        name = (
            "AllVariableMTANDataset"
            if backbone == "MTANQueryFusion"
            else "AllVariablePatchDataset"
        )
    ds = getattr(datasets, name)(
        scaler.transform(data["train_data"]), type="train", global_args=g, **dp
    )
    params = dict(c["model_config"]["model_params"])
    params.update(VARIANTS[method], hid_dim=8, plugin_hidden=8, plugin_heads=2)
    if backbone == "MTANQueryFusion":
        params.update(latent_dim=4, gen_hidden=8, te_dim=8, num_ref_points=8)
    sample = ds[0]
    b = ds.collate_fn([sample])
    m = getattr(models, backbone)(g, **params).eval()
    pred = m(**b)
    changed = copy.deepcopy(sample)
    nodes = changed[0]
    future_y = b["batch_pred_mask"][0].bool()
    nodes[future_y, 1] = float("nan")
    torch.testing.assert_close(pred, m(**ds.collate_fn([changed])))
    assert len(ds.packet.get("evaluation_target_ids", ds.packet["target_ids"])) == 9
    assert len(ds.packet["target_ids"]) == (18 if method == "original" else 9)
    assert ds.npatch == 120
    assert b["batch_timestamp"].max() <= 1
    pred.square().mean().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters()
    )


def test_dist_cli_all_fifteen_and_equal_metric_targets(tmp_path):
    path = tmp_path / "DIST.npz"
    archive(path)
    output = tmp_path / "runs"
    run = subprocess.run(
        [
            sys.executable,
            "scripts/run_query_fusion.py",
            "--output",
            str(output),
            "--data-root",
            str(tmp_path / "unused_tpatch"),
            "--dist-data",
            str(path),
            "--datasets",
            "dist",
            "--backbones",
            *BACKBONES,
            "--variants",
            *METHODS,
            "--epochs",
            "1",
            "--max-samples-per-split",
            "1",
            "--small-model",
            "--batch-size",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    rows = json.loads((output / "summary.json").read_text())
    assert len(rows) == 15 and all(r["exit_code"] == 0 for r in rows)
    for backbone in BACKBONES:
        selected = []
        for method in METHODS:
            result = output / backbone / "dist/3407" / method / "results"
            report = json.loads((result / "metrics.json").read_text())
            p = report["protocol"]
            assert p["target_ids"] == list(range(9))
            assert (
                p["normalization"]
                == "MinMax_on_training_windows_only_no_validation_or_test"
            )
            assert p["history_seconds"] == 7200 and p["forecast_seconds"] == 1800
            mask = np.load(result / "test_y_mask.npy").astype(bool)
            ids = np.load(result / "test_var_idx.npy")[mask]
            truth = np.load(result / "test_y_true.npy")[mask]
            selected.append((ids, truth))
        for ids, truth in selected[1:]:
            np.testing.assert_array_equal(ids, selected[0][0])
            np.testing.assert_array_equal(truth, selected[0][1])


@pytest.mark.parametrize("single", [False, True])
def test_dist_shell_config_and_cuda(tmp_path, single):
    path = tmp_path / "DIST.npz"
    archive(path)
    output = tmp_path / "runs"
    script = "run_query_fusion.sh" if single else "run_dist_query_fusion.sh"
    arguments = ["hipatch", "DIST", "strgcn"] if single else []
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        DIST_DATA_PATH=str(path),
        RUN_DIR=str(output),
        DRY_RUN="1",
    )
    env.pop("BATCH_SIZE", None)
    run = subprocess.run(
        ["bash", str(ROOT / "scripts" / script), *arguments, "--device", "cuda:1"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    configs = list(output.glob("*/dist/*/*/config.yaml"))
    assert len(configs) == (1 if single else 15)
    for path in configs:
        c = yaml.safe_load(path.read_text())
        assert c["train_config"]["batch_size"] == 1
        assert c["data_config"]["data_processor_name"] == "ExEnProcessedProcessor"
    assert (
        json.loads((output / "manifest.json").read_text())["arguments"]["device"]
        == "cuda:1"
    )
