"""Metadata coalescing and optional batch timing without model changes."""
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch
import yaml

from models.STRGCN.RollingWindows import pack_rolling_attention_plan
from test_within_variable_attention import observations

ROOT = Path(__file__).resolve().parents[1]


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        torch.testing.assert_close(a, b)
    elif is_dataclass(a):
        assert type(a) is type(b)
        for field in fields(a): assert_tree_equal(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_tree_equal(x, y)
    else:
        assert a == b


def test_whole_plan_is_transferred_in_three_dtype_slabs_and_keeps_metadata():
    b = observations()
    plan = pack_rolling_attention_plan(b['batch_timestamp'], b['batch_var_idx'],
        b['batch_node_type'], b['batch_pad_mask'].bool(), .2,
        bounds=torch.tensor([[0., .6, 1.]]), within_relations=(3, 4),
        receiver_chunk=2, max_score_elements=32)
    original_to = torch.Tensor.to
    calls = []
    def record(tensor, *args, **kwargs):
        calls.append((tensor.dtype, tensor.ndim))
        return original_to(tensor, *args, **kwargs)
    with patch.object(torch.Tensor, 'to', record):
        moved = plan.to('cpu', non_blocking=True)
    assert len(calls) == 3
    assert {dtype for dtype, ndim in calls} == {torch.int64, torch.bool, torch.float32}
    assert all(ndim == 1 for dtype, ndim in calls)
    assert_tree_equal(plan, moved)
    # Modifying the CPU source after transfer must not alter the copied plan.
    before = moved.windows[0].bounds.clone()
    plan.windows[0].bounds.add_(100)
    torch.testing.assert_close(moved.windows[0].bounds, before)


def test_cpu_profile_reports_first_batch_only_and_preserves_training(tmp_path):
    from test_exen_processed import archive
    path, _, _ = archive(tmp_path)
    c = yaml.safe_load((ROOT / 'config/ExEnGraphTransformer/processed/DIST.yaml').read_text())
    c['global_config']['exen_data_path'] = str(path)
    c['data_config']['data_processor_params'].update(history_len=7, forecast_len=3,
        target_names=['Y'], max_windows_per_split=2)
    c['model_config']['model_params'].update(hid_dim=8, num_heads=2, n_layer=1, n_cross_layer=1)
    c['train_config']['trainer_params']['max_epoch_num'] = 1
    config = tmp_path / 'run.yaml'; config.write_text(yaml.safe_dump(c))
    run = subprocess.run([sys.executable, 'main.py', '--config_path', str(config),
        '--device', 'cpu', '--profile-steps', '1', '--model_save_dir_path', str(tmp_path / 'model'),
        '--result_save_dir_path', str(tmp_path / 'result')], cwd=ROOT, capture_output=True,
        text=True, timeout=90, env=dict(os.environ, OMP_NUM_THREADS='1'))
    assert run.returncode == 0, run.stdout + run.stderr
    timing = [line.split('BATCH_TIMING ', 1)[1] for line in run.stdout.splitlines() if 'BATCH_TIMING ' in line]
    assert len(timing) == 1
    values = json.loads(timing[0])
    assert set(values) == {'data', 'transfer', 'forward', 'backward', 'optimizer', 'total'}
    assert all(value >= 0 for value in values.values())
    assert abs(values['total'] - sum(value for key, value in values.items() if key != 'total')) < 1e-6
    assert (tmp_path / 'result/metrics.json').exists()
