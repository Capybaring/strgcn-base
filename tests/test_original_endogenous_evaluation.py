"""Native network parity, all-variable training and metric-only Y selection."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data_processors import TPatchRecordProcessor
from datasets import AllVariablePatchDataset, AllVariableMTANDataset
from datasets import QueryFusionDataset
from models.exogenous.adapter import TPatchQueryFusion, HiPatchQueryFusion
from models.exogenous.mtan_adapter import MTANQueryFusion
from trainers.TPatchForecastTrainer import TPatchForecastTrainer
from utils.scaler import TPatchInputScaler
from utils.tpatch_evaluation import variable_macro_loss
from test_tpatch_records import make_data

ROOT = Path(__file__).resolve().parents[1]


def setup(root, dataset, backbone):
    make_data(root, dataset)
    g = {"device": "cpu"}
    data = TPatchRecordProcessor(
        g, dataset, data_root=root, max_samples_per_split=2
    ).process()
    scaler = TPatchInputScaler()
    scaler.fit(data["seen_data"])
    packet = scaler.transform(data["train_data"])
    cls = AllVariableMTANDataset if backbone == "mtand" else AllVariablePatchDataset
    ds = cls(
        packet,
        type="train",
        global_args=g,
        patch_size=dict(activity=300, mimic=8, physionet=8, ushcn=2)[dataset],
    )
    params = dict(hid_dim=8, n_heads=1, fusion_mode="none")
    if backbone == "mtand":
        cls = MTANQueryFusion
        params.update(latent_dim=4, gen_hidden=8, te_dim=8, num_ref_points=8)
    else:
        cls = TPatchQueryFusion if backbone == "tpatchgnn" else HiPatchQueryFusion
        params["n_layer"] = 1
    model = cls(g, **params).eval()
    return g, data, packet, scaler, ds, model


@pytest.mark.parametrize("dataset", ["activity", "mimic", "physionet", "ushcn"])
@pytest.mark.parametrize("backbone", ["tpatchgnn", "hipatch", "mtand"])
def test_native_all_variable_parity_and_target_separation(tmp_path, dataset, backbone):
    torch.manual_seed(7)
    g, data, packet, scaler, ds, model = setup(tmp_path, dataset, backbone)
    dim = len(packet["names"])
    assert packet["target_ids"] == data["train_data"]["target_ids"]
    assert ds.packet["target_ids"] == list(range(dim))
    assert ds.packet["evaluation_target_ids"] == packet["target_ids"]
    assert ds.packet["windows"] is packet["windows"]
    b = ds.collate_fn([ds[0], ds[1]])
    # Same Y values/times/windows as the plugin task, despite extra X labels.
    reference = QueryFusionDataset(
        packet,
        type="train",
        global_args={"device": "cpu"},
        patch_size=dict(activity=300, mimic=8, physionet=8, ushcn=2)[dataset],
    )
    ref = reference.collate_fn([reference[0], reference[1]])
    for row in range(2):
        take = b["batch_pred_mask"][row].bool() & torch.isin(
            b["batch_var_idx"][row], torch.tensor(packet["target_ids"])
        )
        other = ref["batch_pred_mask"][row].bool()
        for key in ("batch_value", "batch_timestamp", "batch_var_idx"):
            torch.testing.assert_close(b[key][row, take], ref[key][row, other])
    # All variables occupy their ORIGINAL column indices, without Y-first reorder.
    assert ds.backbone_ids == list(range(dim))
    key = "batch_backbone_mtan_mask" if backbone == "mtand" else "batch_backbone_mask"
    assert b[key][..., packet["exogenous_ids"]].any()
    mask = b["batch_pred_mask"].bool() & b["batch_pad_mask"].bool()
    assert torch.isin(
        b["batch_var_idx"][mask], torch.tensor(packet["exogenous_ids"])
    ).any()
    features = model.encode_queries(b)
    if backbone == "mtand":
        native = model.core(
            b["batch_backbone_mtan_values"],
            b["batch_backbone_mtan_times"],
            b["batch_backbone_mtan_mask"],
            b["batch_backbone_queries"],
            sample=False,
        )
    else:
        native = model.core.forecasting(
            b["batch_backbone_queries"],
            b["batch_backbone_values"],
            b["batch_backbone_times"],
            b["batch_backbone_mask"],
        )[0]
    torch.testing.assert_close(model.decode(features), native)
    assert not hasattr(model, "fusion")
    before = model(**b)
    altered = copy.deepcopy(b)
    altered["batch_value"][mask] = float("nan")
    # Future values are labels of ALL variables, never backbone inputs.
    torch.testing.assert_close(before, model(**altered))
    pred = before.squeeze(-1)
    loss = variable_macro_loss(pred, b["batch_value"], mask, b["batch_var_idx"], dim)
    direct = []
    for var in range(dim):
        take = mask & (b["batch_var_idx"] == var)
        if take.any():
            direct.append((pred[take] - b["batch_value"][take]).square().mean())
    torch.testing.assert_close(loss, torch.stack(direct).mean())
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.core.parameters()
    )


def test_eval_scope_changes_only_mask_and_metrics_not_training(tmp_path):
    g, _, packet, scaler, ds, model = setup(tmp_path, "activity", "tpatchgnn")
    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn, shuffle=False)
    trainer = TPatchForecastTrainer(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        None,
        scaler,
        str(tmp_path / "states"),
        str(tmp_path / "results"),
        max_epoch_num=1,
        do_inverse_scaling=False,
        global_args=g,
    )
    model_before = copy.deepcopy(model.state_dict())
    b = next(iter(loader))
    original_mask = b["batch_pred_mask"].clone()
    pred_before = trainer._predict(b)
    train_loss = variable_macro_loss(
        pred_before,
        b["batch_value"],
        original_mask.bool(),
        b["batch_var_idx"],
        g["num_of_vals"],
    )
    trainer.evaluate(loader, ["mse", "mae", "rmse"], test_flag=True)
    mask_y = trainer.last_eval_mask.copy()
    summary_y = dict(trainer.last_eval_summary)
    assert summary_y["target_ids"] == packet["target_ids"]
    expected_mask = np.isin(trainer.last_eval_var_idx, packet["target_ids"])
    assert not mask_y[~expected_mask].any()
    trainer.evaluation_scope = "all"
    _, _, pred, truth = trainer.evaluate(loader, ["mse", "mae", "rmse"], test_flag=True)
    mask_all = trainer.last_eval_mask.copy()
    assert mask_all.sum() > mask_y.sum()
    ids = trainer.last_eval_var_idx
    errors = [
        np.mean((pred[mask_y & (ids == var)] - truth[mask_y & (ids == var)]) ** 2)
        for var in packet["target_ids"]
    ]
    np.testing.assert_allclose(summary_y["mse"], np.mean(errors), rtol=1e-6)
    torch.testing.assert_close(b["batch_pred_mask"], original_mask)
    torch.testing.assert_close(pred_before, trainer._predict(b))
    torch.testing.assert_close(
        train_loss,
        variable_macro_loss(
            trainer._predict(b),
            b["batch_value"],
            original_mask.bool(),
            b["batch_var_idx"],
            g["num_of_vals"],
        ),
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, model_before[key])


def test_original_batch_plans_all_twelve_without_plugins(tmp_path):
    import yaml

    data, output = tmp_path / "data", tmp_path / "runs"
    for dataset in ("activity", "mimic", "physionet", "ushcn"):
        (data / dataset / "processed").mkdir(parents=True)
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        TPATCH_DATA_ROOT=str(data),
        RUN_DIR=str(output),
        DRY_RUN="1",
    )
    command = ["bash", str(ROOT / "scripts/run_original_backbones.sh")]
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr
    configs = list(output.glob("*/*/3407/original/config.yaml"))
    assert len(configs) == 12
    for path in configs:
        c = yaml.safe_load(path.read_text())
        assert c["model_config"]["model_params"]["fusion_mode"] == "none"
        assert c["data_config"]["dataset_params"]["backbone_history"] == "yx"
        assert (
            c["data_config"]["dataset_params"]["known_future_exogenous_var_ids"] == []
        )
        assert c["train_config"]["trainer_params"]["evaluation_scope"] == "endogenous"
    again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert again.returncode != 0 and "FileExistsError" in again.stderr


def test_original_cli_one_epoch_all_models_and_datasets(tmp_path):
    data, output = tmp_path / "data", tmp_path / "runs"
    for dataset in ("activity", "mimic", "physionet", "ushcn"):
        make_data(data, dataset)
    run = subprocess.run(
        [
            sys.executable,
            "scripts/run_query_fusion.py",
            "--output",
            str(output),
            "--data-root",
            str(data),
            "--variants",
            "original",
            "--seeds",
            "3407",
            "--backbones",
            "TPatchQueryFusion",
            "HiPatchQueryFusion",
            "MTANQueryFusion",
            "--epochs",
            "1",
            "--max-samples-per-split",
            "2",
            "--small-model",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    rows = json.loads((output / "summary.json").read_text())
    assert len(rows) == 12 and all(row["exit_code"] == 0 for row in rows)
    for path in output.glob("*/*/3407/original/results"):
        m = json.loads((path / "metrics.json").read_text())
        p = m["protocol"]
        assert len(p["training_target_ids"]) > len(p["target_ids"])
        assert (
            p["fusion"]["mode"] == "none"
            and p["forecast_inputs"] == "all_variable_history_only"
        )
        mask = np.load(path / "test_y_mask.npy")
        ids = np.load(path / "test_var_idx.npy")
        assert set(ids[mask].tolist()) <= set(p["target_ids"])
        assert not np.isin(ids[mask], p["exogenous_var_ids"]).any()
