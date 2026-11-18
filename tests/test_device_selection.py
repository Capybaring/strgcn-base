"""Explicit CUDA ordinals must reach child processes without an implicit GPU0 mask."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def launch(tmp_path, arguments, environment=None, script="run_query_fusion.sh"):
    data = tmp_path / "data"
    for dataset in ("activity", "mimic", "physionet", "ushcn"):
        (data / dataset / "processed").mkdir(parents=True, exist_ok=True)
    output = tmp_path / "runs"
    env = dict(os.environ, PYTHON=sys.executable, DRY_RUN="1",
               TPATCH_DATA_ROOT=str(data), RUN_DIR=str(output))
    for key in ("GPU", "DEVICE", "CUDA_VISIBLE_DEVICES", "BATCH_SIZE"):
        env.pop(key, None)
    env.update(environment or {})
    result = subprocess.run(["bash", str(ROOT / "scripts" / script), *arguments],
                            env=env, capture_output=True, text=True, timeout=20)
    return result, output


@pytest.mark.parametrize("arguments,environment,device,visibility", [
    (["hipatch", "mimic", "original", "--device", "cuda:1"], {}, "cuda:1", None),
    (["--device=cuda:2", "hipatch", "mimic", "original"], {}, "cuda:2", None),
    (["hipatch", "mimic", "original", "--device", "cpu"], {}, "cpu", None),
    (["hipatch", "mimic", "original"], {"DEVICE": "cuda:1"}, "cuda:1", None),
    (["hipatch", "mimic", "original"], {"GPU": "1"}, "cuda:0", "1"),
    (["hipatch", "mimic", "original"], {}, "cuda:0", "0"),
    (["hipatch", "mimic", "original", "--device", "cuda:1"], {"GPU": "0"}, "cuda:1", None),
    (["hipatch", "mimic", "original", "--device", "cuda:1"], {"CUDA_VISIBLE_DEVICES": "2,3"}, "cuda:1", "2,3"),
])
def test_single_device_selection_and_unchanged_native_config(tmp_path, arguments, environment, device, visibility):
    result, output = launch(tmp_path, arguments, environment)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["arguments"]["device"] == device
    assert manifest["cuda_visible_devices"] == visibility
    config = yaml.safe_load(next(output.glob("*/*/*/*/config.yaml")).read_text())
    source = yaml.safe_load((ROOT / "config/HiPatchQueryFusion/tpatch/mimic.yaml").read_text())
    assert config["train_config"]["batch_size"] == source["train_config"]["batch_size"] == 32
    assert config["model_config"]["model_params"] == dict(source["model_config"]["model_params"], fusion_mode="none")


@pytest.mark.parametrize("script,arguments,count", [
    ("run_original_backbones.sh", ["hipatch", "--device", "cuda:1"], 4),
    ("run_hipatch_base_mlp_v1.sh", ["--device", "cuda:1"], 12),
    ("run_mtand_base_mlp_v1.sh", ["--device=cuda:2"], 12),
    ("run_strgcn_query_fusion.sh", ["--device", "cuda:1", "hipatch"], 16),
])
def test_batch_launchers_accept_cuda_device(tmp_path, script, arguments, count):
    result, output = launch(tmp_path, arguments, script=script)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["arguments"]["device"] in ("cuda:1", "cuda:2")
    assert manifest["cuda_visible_devices"] is None
    assert len(list(output.glob("*/*/*/*/config.yaml"))) == count


@pytest.mark.parametrize("arguments", [["--device"], ["--device", "cuda:-1"], ["--device=foo"], ["--device="]])
def test_invalid_device_fails_before_run_creation(tmp_path, arguments):
    result, output = launch(tmp_path, arguments)
    assert result.returncode == 2 and not output.exists()


@pytest.mark.parametrize("device,distributed,expected", [
    ("cuda:1", False, 1), ("cuda:0", False, 0), ("cuda", False, 2),
    ("cpu", False, None), ("cuda:1", True, None),
])
def test_current_cuda_device_binding_without_gpu(device, distributed, expected):
    from main import select_cuda_device

    with patch("torch.cuda.set_device") as select, patch("torch.cuda.current_device", return_value=2):
        select_cuda_device(device, distributed)
        if expected is None:
            select.assert_not_called()
        else:
            select.assert_called_once_with(expected)
