"""The simple shell entry must never expand a single experiment into a matrix."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_query_fusion.sh"


@pytest.mark.parametrize("dedicated,batch,gpu", [(False, "1", "1"), (True, None, "1"), (True, "2", "3")])
def test_hipatch_mimic_batch_override_preserves_native_config(tmp_path, dedicated, batch, gpu):
    import yaml

    data, output = tmp_path / "data", tmp_path / "runs"
    (data / "mimic" / "processed").mkdir(parents=True)
    env = dict(os.environ, PYTHON=sys.executable, TPATCH_DATA_ROOT=str(data),
               RUN_DIR=str(output), DRY_RUN="1", GPU=gpu)
    env.pop("BATCH_SIZE", None)
    if batch is not None:
        env["BATCH_SIZE"] = batch
    command = (["bash", str(ROOT / "scripts/run_hipatch_mimic_original.sh")] if dedicated
               else ["bash", str(SCRIPT), "hipatch", "mimic", "original"])
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr
    path = output / "HiPatchQueryFusion/mimic/3407/original/config.yaml"
    config = yaml.safe_load(path.read_text())
    reference = yaml.safe_load((ROOT / "config/HiPatchQueryFusion/tpatch/mimic.yaml").read_text())
    assert config["train_config"]["batch_size"] == int(batch or "1")
    params = dict(reference["model_config"]["model_params"], fusion_mode="none")
    assert config["model_config"]["model_params"] == params
    assert config["data_config"]["dataset_params"]["patch_size"] == 12
    assert config["train_config"]["trainer_params"]["evaluation_scope"] == "endogenous"
    assert f"Physical GPU(s): {gpu}; process device: cuda:0" in run.stdout
    assert json.loads((output / "manifest.json").read_text())["arguments"]["batch_size"] == int(batch or "1")


def test_rejects_zero_batch_before_creating_output(tmp_path):
    output = tmp_path / "must not exist"
    run = subprocess.run([sys.executable, str(ROOT / "scripts/run_query_fusion.py"),
                          "--output", str(output), "--data-root", str(tmp_path),
                          "--batch-size", "0", "--plan-only"],
                         capture_output=True, text=True, timeout=20)
    assert run.returncode != 0 and "--batch-size must be positive" in run.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "arguments,model,dataset,method",
    [
        ([], "TPatchQueryFusion", "activity", "base"),
        (["tpatchgnn", "activity", "v1"], "TPatchQueryFusion", "activity", "v1"),
        (["hipatch", "ushcn", "v2"], "HiPatchQueryFusion", "ushcn", "v2"),
        (["mtand", "activity", "v1"], "MTANQueryFusion", "activity", "v1"),
        (["hipatch", "activity", "strgcn"], "HiPatchQueryFusion", "activity", "strgcn"),
        (["mtand", "activity", "v1_all"], "MTANQueryFusion", "activity", "v1_all"),
    ],
)
def test_one_invocation_creates_one_full_size_fixed_seed_config(
    tmp_path,
    arguments,
    model,
    dataset,
    method,
):
    data = tmp_path / "data with spaces"
    (data / dataset / "processed").mkdir(parents=True)
    output = tmp_path / "one run"
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        TPATCH_DATA_ROOT=str(data),
        RUN_DIR=str(output),
        DRY_RUN="1",
    )
    run = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    configs = list(output.glob("*/*/*/*/config.yaml"))
    assert configs == [output / model / dataset / "3407" / method / "config.yaml"]
    args = json.loads((output / "manifest.json").read_text())["arguments"]
    assert args["backbones"] == [model] and args["datasets"] == [dataset]
    assert args["variants"] == [method] and args["seeds"] == [3407]
    assert args["epochs"] == 30 and not args["small_model"]
    assert args["max_samples_per_split"] is None
    assert args["stream_logs"] and args["plan_only"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["all", "activity", "v1"],
        ["tpatchgnn", "all", "v1"],
        ["tpatchgnn", "activity", "all"],
        ["tpatchgnn", "activity", "v1", "v2"],
    ],
)
def test_rejects_multiple_or_invalid_experiments(tmp_path, arguments):
    output = tmp_path / "must not exist"
    run = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=tmp_path,
        env=dict(os.environ, RUN_DIR=str(output)),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run.returncode == 2 and not output.exists()


@pytest.mark.parametrize(
    "backbone,model_name",
    [("hipatch", "HiPatchQueryFusion"), ("mtand", "MTANQueryFusion")],
)
def test_batch_contains_exactly_four_datasets_and_three_methods(
    tmp_path, backbone, model_name
):
    data = tmp_path / "data with spaces"
    datasets = ("activity", "mimic", "physionet", "ushcn")
    methods = ("base", "mlp", "v1")
    for dataset in datasets:
        (data / dataset / "processed").mkdir(parents=True)
    output = tmp_path / "twelve runs"
    command = ["bash", str(ROOT / f"scripts/run_{backbone}_base_mlp_v1.sh")]
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        TPATCH_DATA_ROOT=str(data),
        RUN_DIR=str(output),
        DRY_RUN="1",
    )
    run = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20
    )
    assert run.returncode == 0, run.stdout + run.stderr
    expected = {
        output / model_name / dataset / "3407" / method / "config.yaml"
        for dataset in datasets
        for method in methods
    }
    assert set(output.glob("*/*/*/*/config.yaml")) == expected
    args = json.loads((output / "manifest.json").read_text())["arguments"]
    assert args["backbones"] == [model_name]
    assert args["datasets"] == list(datasets) and args["variants"] == list(methods)
    assert args["seeds"] == [3407] and args["epochs"] == 30
    assert not args["small_model"] and args["max_samples_per_split"] is None
    assert args["stream_logs"] and args["input_mode"] == "FullExo"
    # The batch entry must preserve the runner's refusal to overwrite old runs.
    again = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20
    )
    assert again.returncode != 0 and "FileExistsError" in again.stderr


@pytest.mark.parametrize(
    "backbone,model",
    [
        ("hipatch", "HiPatchQueryFusion"),
        ("tpatchgnn", "TPatchQueryFusion"),
        ("mtand", "MTANQueryFusion"),
    ],
)
def test_strgcn_batch_visibility_and_encoder_configs(tmp_path, backbone, model):
    import yaml

    data, output = tmp_path / "data with spaces", tmp_path / "runs"
    for dataset in ("activity", "mimic", "physionet", "ushcn"):
        (data / dataset / "processed").mkdir(parents=True)
    run = subprocess.run(
        ["bash", str(ROOT / "scripts/run_strgcn_query_fusion.sh"), backbone],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHON=sys.executable,
            TPATCH_DATA_ROOT=str(data),
            RUN_DIR=str(output),
            DRY_RUN="1",
        ),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    configs = list(output.glob("*/*/*/*/config.yaml"))
    assert len(configs) == 16
    for path in configs:
        assert path.parts[-5] == model and path.parts[-3] == "3407"
        method = path.parent.name
        params = yaml.safe_load(path.read_text())["model_config"]["model_params"]
        if method in ("strgcn", "v1_all"):
            assert params["allow_future_x"] is True
            assert params.get("x_encoder", "pointwise") == (
                "strgcn" if method == "strgcn" else "pointwise"
            )
