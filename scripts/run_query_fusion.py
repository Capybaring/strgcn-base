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
    "original": dict(fusion_mode="none"),
    "mlp": dict(fusion_mode="mlp"),
    "cross": dict(fusion_mode="cross"),
    "v1": dict(fusion_mode="v1"),
    "v1_all": dict(fusion_mode="v1", allow_future_x=True),
    "strgcn": dict(fusion_mode="v1", x_encoder="strgcn", allow_future_x=True),
    "graph": dict(
        fusion_mode="v1",
        x_encoder="irgraph",
        allow_future_x=True,
        use_gap_encoding=False,
        use_density_encoding=False,
    ),
    "irgraph": dict(fusion_mode="v1", x_encoder="irgraph", allow_future_x=True),
    "voronoi": dict(fusion_mode="v1", x_encoder="voronoi", allow_future_x=True),
    "voronoi_star": dict(
        fusion_mode="v1",
        x_encoder="voronoi_star",
        allow_future_x=True,
        x_layers=2,
    ),
    "voronoi_star_no_density": dict(
        fusion_mode="v1",
        x_encoder="voronoi_star",
        allow_future_x=True,
        use_density_encoding=False,
        x_layers=2,
    ),
    "voronoi_h": dict(
        fusion_mode="v1",
        x_encoder="voronoi",
        allow_future_x=True,
        density_location="node",
    ),
    "voronoi_h_no_density": dict(
        fusion_mode="v1",
        x_encoder="voronoi",
        allow_future_x=True,
        density_location="node",
        use_density_encoding=False,
    ),
    "voronoi_h_4d": dict(
        fusion_mode="v1",
        x_encoder="voronoi",
        allow_future_x=True,
        density_location="node",
        node_encoding="concat",
    ),
    "voronoi_h_4d_no_density": dict(
        fusion_mode="v1",
        x_encoder="voronoi",
        allow_future_x=True,
        density_location="node",
        node_encoding="concat",
        use_density_encoding=False,
    ),
    "voronoi_no_density": dict(
        fusion_mode="v1",
        x_encoder="voronoi",
        allow_future_x=True,
        use_density_encoding=False,
    ),
    "irgraph_no_node": dict(
        fusion_mode="v1",
        x_encoder="irgraph",
        allow_future_x=True,
        use_gap_encoding=False,
    ),
    "irgraph_no_density": dict(
        fusion_mode="v1",
        x_encoder="irgraph",
        allow_future_x=True,
        use_density_encoding=False,
    ),
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
    p.add_argument("--dist-data", type=Path, default=ROOT / "data/processed/DIST.npz")
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["activity", "mimic", "physionet", "ushcn", "dist"],
        default=["activity", "mimic", "physionet", "ushcn"],
    )
    p.add_argument(
        "--backbones",
        nargs="+",
        choices=[
            "TPatchQueryFusion",
            "HiPatchQueryFusion",
            "MTANQueryFusion",
            "GraFITiQueryFusion",
            "CRUQueryFusion",
            "LatentODEQueryFusion",
        ],
        default=["TPatchQueryFusion", "HiPatchQueryFusion"],
    )
    p.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[3407])
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--input-mode", choices=["FullExo", "HistExo", "NoExo"], default="FullExo"
    )
    p.add_argument("--backbone-history", choices=["y", "yx"], default="y")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--batch-size", type=int, help="Override only training/evaluation batch size"
    )
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
    if a.batch_size is not None and a.batch_size < 1:
        p.error("--batch-size must be positive")
    if "original" in a.variants and a.input_mode != "FullExo":
        p.error("original preserves all variable histories; use --input-mode FullExo")
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
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
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
                    config_source = (
                        ROOT / f"config/{backbone}/processed/DIST.yaml"
                        if dataset == "dist"
                        else ROOT / f"config/{backbone}/tpatch/{dataset}.yaml"
                    )
                    c = yaml.safe_load(config_source.read_text())
                    if dataset == "dist":
                        c["global_config"]["exen_data_path"] = str(
                            a.dist_data.resolve()
                        )
                    else:
                        c["global_config"]["tpatch_data_root"] = str(data_root)
                        c["data_config"]["data_processor_params"]["data_root"] = str(
                            data_root
                        )
                    if a.max_samples_per_split:
                        c["data_config"]["data_processor_params"][
                            "max_windows_per_split"
                            if dataset == "dist"
                            else "max_samples_per_split"
                        ] = a.max_samples_per_split
                    dp = c["data_config"]["dataset_params"]
                    dp["backbone_history"] = a.backbone_history
                    if a.input_mode == "NoExo":
                        dp["use_exogenous"] = False
                    if a.input_mode == "HistExo":
                        dp["known_future_exogenous_var_ids"] = []
                    c["model_config"]["model_params"].update(VARIANTS[variant])
                    if variant == "original":
                        c["data_config"]["dataset_name"] = (
                            "AllVariableMTANDataset"
                            if backbone
                            in (
                                "MTANQueryFusion",
                                "GraFITiQueryFusion",
                                "CRUQueryFusion",
                                "LatentODEQueryFusion",
                            )
                            else "AllVariablePatchDataset"
                        )
                        dp.update(
                            backbone_history="yx",
                            known_future_exogenous_var_ids=[],
                            use_exogenous=True,
                        )
                        c["train_config"]["trainer_params"]["evaluation_scope"] = (
                            "endogenous"
                        )
                    if a.small_model:
                        small = dict(
                            hid_dim=8, n_heads=1, plugin_hidden=8, plugin_heads=2
                        )
                        if backbone == "MTANQueryFusion":
                            small.update(
                                latent_dim=4, gen_hidden=8, te_dim=8, num_ref_points=8
                            )
                        elif backbone in (
                            "GraFITiQueryFusion",
                            "CRUQueryFusion",
                            "LatentODEQueryFusion",
                        ):
                            small.update(
                                latent_dim=4,
                                n_layer=1,
                                ode_hidden=8,
                                gru_units=8,
                                num_basis=2,
                                bandwidth=1,
                            )
                        else:
                            small["n_layer"] = 1
                        c["model_config"]["model_params"].update(small)
                        c["train_config"]["batch_size"] = 2
                    if a.batch_size is not None:
                        c["train_config"]["batch_size"] = a.batch_size
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
