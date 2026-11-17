#!/usr/bin/env python3
"""Reproducible two-backbone matrix; a fresh run directory is mandatory."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "base": dict(fusion_mode="none"),
    "mlp": dict(fusion_mode="mlp"),
    "cross": dict(fusion_mode="cross"),
    "v1": dict(fusion_mode="v1"),
    "v2": dict(fusion_mode="v2"),
    "v2_static": dict(fusion_mode="v2", transfer="static"),
    "v2_no_transfer": dict(fusion_mode="v2", transfer="none"),
    "v1_no_time": dict(fusion_mode="v1", use_time_bias=False),
    "v1_no_variable": dict(fusion_mode="v1", use_variable_bias=False),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["activity", "mimic", "physionet", "ushcn"],
        default=["activity", "mimic", "physionet", "ushcn"],
    )
    p.add_argument(
        "--backbones",
        nargs="+",
        choices=["TPatchQueryFusion", "HiPatchQueryFusion", "MTANQueryFusion"],
        default=["TPatchQueryFusion", "HiPatchQueryFusion"],
    )
    p.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--input-mode", choices=["FullExo", "HistExo", "NoExo"], default="FullExo"
    )
    p.add_argument("--backbone-history", choices=["y", "yx"], default="y")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-samples-per-split", type=int)
    p.add_argument(
        "--small-model",
        action="store_true",
        help="diagnostic only; hidden=8, one encoder layer",
    )
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--stream-logs",
        action="store_true",
        help="Display training progress while saving the same output to run.log",
    )
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data_root = a.data_root.resolve()
    manifest = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dirty=subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        diagnostic=bool(a.small_model or a.max_samples_per_split or a.epochs < 30),
        python=sys.version,
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    for backbone in a.backbones:
        for dataset in a.datasets:
            for seed in a.seeds:
                for variant in a.variants:
                    run = out / backbone / dataset / str(seed) / variant
                    run.mkdir(parents=True)
                    c = yaml.safe_load(
                        (ROOT / f"config/{backbone}/tpatch/{dataset}.yaml").read_text()
                    )
                    c["global_config"]["tpatch_data_root"] = str(data_root)
                    c["data_config"]["data_processor_params"]["data_root"] = str(
                        data_root
                    )
                    if a.max_samples_per_split:
                        c["data_config"]["data_processor_params"][
                            "max_samples_per_split"
                        ] = a.max_samples_per_split
                    dp = c["data_config"]["dataset_params"]
                    dp["backbone_history"] = a.backbone_history
                    if a.input_mode == "NoExo":
                        dp["use_exogenous"] = False
                    if a.input_mode == "HistExo":
                        dp["known_future_exogenous_var_ids"] = []
                    c["model_config"]["model_params"].update(VARIANTS[variant])
                    if a.small_model:
                        small = dict(
                            hid_dim=8, n_heads=1, plugin_hidden=8, plugin_heads=2
                        )
                        if backbone == "MTANQueryFusion":
                            small.update(
                                latent_dim=4, gen_hidden=8, te_dim=8, num_ref_points=8
                            )
                        else:
                            small["n_layer"] = 1
                        c["model_config"]["model_params"].update(small)
                        c["train_config"]["batch_size"] = 2
                    c["train_config"]["random_seed"] = seed
                    c["train_config"]["trainer_params"]["max_epoch_num"] = a.epochs
                    config = run / "config.yaml"
                    config.write_text(yaml.safe_dump(c, sort_keys=False))
                    if a.plan_only:
                        continue
                    cmd = [
                        sys.executable,
                        "-u",
                        "main.py",
                        "--config_path",
                        str(config),
                        "--device",
                        a.device,
                        "--model_save_dir_path",
                        str(run / "checkpoints"),
                        "--result_save_dir_path",
                        str(run / "results"),
                    ]
                    start = time.perf_counter()
                    with (run / "run.log").open("w") as log:
                        env = {
                            **os.environ,
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                        }
                        if a.stream_logs:
                            with subprocess.Popen(
                                cmd,
                                cwd=ROOT,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                env=env,
                                text=True,
                                bufsize=1,
                            ) as process:
                                for line in process.stdout:
                                    log.write(line)
                                    log.flush()
                                    print(line, end="", flush=True)
                                returncode = process.wait()
                        else:
                            returncode = subprocess.run(
                                cmd,
                                cwd=ROOT,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                env=env,
                            ).returncode
                    elapsed = time.perf_counter() - start
                    row = dict(
                        backbone=backbone,
                        dataset=dataset,
                        seed=seed,
                        variant=variant,
                        exit_code=returncode,
                        wall_seconds=elapsed,
                    )
                    if returncode == 0:
                        metrics = json.loads((run / "results/metrics.json").read_text())
                        row.update(
                            mse=metrics["test"]["mse"],
                            mae=metrics["test"]["mae"],
                            parameters=metrics["parameter_count"],
                            epochs=len(metrics["training_history"]),
                            train_seconds=sum(metrics.get("epoch_train_seconds", [])),
                        )
                    rows.append(row)
                    (out / "summary.json").write_text(json.dumps(rows, indent=2))
                    print(json.dumps(row), flush=True)
    if rows:
        with (out / "summary.csv").open("w") as f:
            fields = [
                "backbone",
                "dataset",
                "seed",
                "variant",
                "exit_code",
                "wall_seconds",
                "mse",
                "mae",
                "parameters",
                "epochs",
                "train_seconds",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return int(any(row["exit_code"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
