"""Run with python -m baseline.dag_irregular.run --help from repository root."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from data_processors.ExEnProcessedProcessor import ExEnProcessedProcessor
from .dataset import DAGProcessedDataset
from utils.scaler import ExEnProcessedMinMaxScaler
from utils.dag_evaluation import DAGEvaluationScaler, METRICS
from .adapter import IrregularDAG
from .trainer import DAGIrregularTrainer


def run(args):
    config = yaml.safe_load(Path(args.config).read_text())
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    global_args = dict(config['global_config'], device=args.device)
    if args.data:
        global_args['exen_data_path'] = args.data
    processor = dict(config['data_config']['data_processor_params'])
    if args.max_windows:
        processor['max_windows_per_split'] = args.max_windows
    packets = ExEnProcessedProcessor(global_args, **processor).process()
    evaluation = DAGEvaluationScaler().fit(packets['train_data'])
    scaler = ExEnProcessedMinMaxScaler(); scaler.fit(packets['train_data'])
    dataset_params = config['data_config']['dataset_params']
    loaders = []
    for split in ('train', 'valid', 'test'):
        ds = DAGProcessedDataset(scaler.transform(packets[f'{split}_data']), type=split,
            global_args=global_args, known_future_exogenous_var_ids=dataset_params.get('known_future_exogenous_var_ids','all'),
            use_exogenous=args.mode == 'FullExo')
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=split == 'train',
                                  collate_fn=ds.collate_fn, drop_last=False))
    p = packets['train_data']
    model_params = dict(config.get('dag', {}))
    if args.tiny:
        model_params.update(seq_len=8, patch_len=4, stride=2, d_model=8, d_ff=16,
                            n_heads=2, e_layers=1, dropout=0.)
    model = IrregularDAG(p['target_ids'], p['exogenous_ids'], p['history_seconds'],
        p['duration_seconds']-p['history_seconds'], use_exogenous=args.mode == 'FullExo',
        **model_params).to(args.device)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    scaler.save(out/'input_scaler'); evaluation.save(out/'evaluation_scaler.npz')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # DAG type3: after epoch e, lr = base for e<3, else base*.9**(e-3).
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda epoch: 1. if epoch < 3 else .9**(epoch-3))
    trainer = DAGIrregularTrainer(model, optimizer, scheduler, scaler, str(out), str(out),
        max_epoch_num=args.epochs, enable_early_stop=True, early_stop_patience=10,
        eval_metric_key='loss', enable_autocast=False, do_inverse_scaling=False,
        global_args=global_args, evaluation_scaler=evaluation)
    best, history = trainer.train(loaders[0], loaders[1], metrics=METRICS)
    result, predicted, actual = trainer.test(loaders[2], metrics=METRICS)
    # Export only actual valid queries with absolute physical timestamps.
    valid = trainer.last_eval_mask.astype(bool)
    query_times = np.concatenate([b['batch_absolute_timestamp'].numpy().reshape(-1) for b in loaders[2]])
    window_ids = trainer.last_eval_window_id
    np.savez(out/'predictions.npz', prediction=predicted[valid], actual=actual[valid],
        variable_id=trainer.last_eval_var_idx[valid], window_id=window_ids[valid],
        timestamp=query_times[valid])
    manifest = dict(mode=args.mode, arguments=vars(args), source_config=config,
        split=global_args['exen_processed_summary'], best_validation=best, test=result,
        evaluation_scaler=evaluation.manifest(), input_scaler='train-only MinMax',
        history_grid_seconds=(model.history_grid.cpu().numpy()*p['duration_seconds']).tolist(),
        future_grid_seconds=(model.future_grid.cpu().numpy()*p['duration_seconds']).tolist(),
        known_future_exogenous_var_ids=sorted(loaders[0].dataset.known_future_exogenous_var_ids),
        target_ids=p['target_ids'], exogenous_ids=p['exogenous_ids'],
        parameter_count=sum(v.numel() for v in model.parameters()),
        objective='MAE(real valid Y queries) + beta*(TC MAE + CC reconstruction MAE)',
        checkpoint_selection='validation query MAE in model input units',
        metric_aggregation='equal window weight; valid real query cells only',
        no_exogenous_compatibility='one constant-zero placeholder; exogenous attention and auxiliary terms disabled' if args.mode=='NoExo' else None)
    (out/'metrics.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(dict(output=str(out), test=result), indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True, help='Existing processed dataset YAML, e.g. config/ExEn/processed/DIST.yaml')
    parser.add_argument('--data', help='Override processed NPZ location')
    parser.add_argument('--mode', choices=['FullExo','NoExo'], default='FullExo')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.02)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-windows', type=int)
    parser.add_argument('--tiny', action='store_true', help='Small model for smoke tests only')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error('epochs, batch-size, threads and lr must be positive')
    run(args)


if __name__ == '__main__':
    main()
