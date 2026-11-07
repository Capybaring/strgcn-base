"""Shared processed-data runner for TimeXer and CrossLinear."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from data_processors.ExEnProcessedProcessor import ExEnProcessedProcessor
from baseline.dag_irregular.dataset import DAGProcessedDataset
from utils.scaler import ExEnProcessedMinMaxScaler
from utils.dag_evaluation import DAGEvaluationScaler,METRICS
from .trainer import GridForecastTrainer

PROFILES = {
    'timexer':dict(loss='huber',lradj='type3',lr=.0001,epochs=100,patience=10,
                   variant='TimeXer-DAG-future-encoder'),
    'crosslinear':dict(loss='mse',lradj='type1',lr=.0001,epochs=100,patience=10,
                       variant='CrossLinear+FutureX-MLP'),
}


def make_parser(method):
    p=argparse.ArgumentParser(description=f'{method} irregular forecasting baseline')
    p.set_defaults(method=method)
    p.add_argument('--config',required=True,help='Existing config/ExEn/processed/*.yaml dataset configuration')
    p.add_argument('--data',help='Override processed NPZ location')
    p.add_argument('--mode',choices=['FullExo','NoExo'],default='FullExo')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=PROFILES[method]['epochs'])
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=PROFILES[method]['lr'])
    p.add_argument('--patience',type=int,default=PROFILES[method]['patience'])
    p.add_argument('--loss',choices=['mse','mae','huber'],default=PROFILES[method]['loss'])
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--seq-len',type=int,help='Override regular history grid size')
    p.add_argument('--max-windows',type=int)
    p.add_argument('--tiny',action='store_true',help='Small model for smoke tests only')
    return p


def run(args):
    if min(args.epochs,args.batch_size,args.threads,args.patience)<1 or not np.isfinite(args.lr) or args.lr<=0:
        raise ValueError('epochs, batch-size, threads, patience and finite lr must be positive')
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}; choose a new output path')
    config=yaml.safe_load(Path(args.config).read_text())
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    global_args=dict(config['global_config'],device=args.device)
    if args.data:global_args['exen_data_path']=args.data
    processor=dict(config['data_config']['data_processor_params'])
    if args.max_windows is not None:processor['max_windows_per_split']=args.max_windows
    packets=ExEnProcessedProcessor(global_args,**processor).process()
    evaluation=DAGEvaluationScaler().fit(packets['train_data'])
    scaler=ExEnProcessedMinMaxScaler();scaler.fit(packets['train_data'])
    dataset_params=config['data_config']['dataset_params']
    loaders=[]
    for split in ('train','valid','test'):
        dataset=DAGProcessedDataset(scaler.transform(packets[f'{split}_data']),type=split,
            global_args=global_args,known_future_exogenous_var_ids=dataset_params.get('known_future_exogenous_var_ids','all'),
            use_exogenous=args.mode=='FullExo')
        loaders.append(DataLoader(dataset,batch_size=args.batch_size,shuffle=split=='train',
                                  collate_fn=dataset.collate_fn,drop_last=False))
    p=packets['train_data']
    params=dict(config.get(args.method,{}))
    if args.seq_len is not None:params['seq_len']=args.seq_len
    if args.tiny:
        params.update(seq_len=8,patch_len=4,d_model=8,d_ff=16)
        if args.method=='timexer':params.update(n_heads=2,e_layers=1,dropout=0.)
        else:params['fusion_hidden']=8
    if args.method=='timexer':
        from baseline.timexer_irregular.adapter import IrregularTimeXer
        model_class=IrregularTimeXer
    else:
        from baseline.crosslinear_irregular.adapter import IrregularCrossLinear
        model_class=IrregularCrossLinear
    model=model_class(p['target_ids'],p['exogenous_ids'],p['history_seconds'],
        p['duration_seconds']-p['history_seconds'],use_exogenous=args.mode=='FullExo',**params).to(args.device)
    out.mkdir(parents=True,exist_ok=True)
    scaler.save(out/'input_scaler');evaluation.save(out/'evaluation_scaler.npz')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    lradj=PROFILES[args.method]['lradj']
    schedule=(lambda epoch: 1. if epoch<3 else .9**(epoch-3)) if lradj=='type3' else (lambda epoch:.5**max(epoch-1,0))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    trainer=GridForecastTrainer(model,optimizer,scheduler,scaler,str(out),str(out),
        objective=args.loss,max_epoch_num=args.epochs,enable_early_stop=True,
        early_stop_patience=args.patience,eval_metric_key='loss',enable_autocast=False,
        do_inverse_scaling=False,global_args=global_args,evaluation_scaler=evaluation)
    best,history=trainer.train(loaders[0],loaders[1],metrics=METRICS)
    result,predicted,actual=trainer.test(loaders[2],metrics=METRICS)
    valid=trainer.last_eval_mask.astype(bool)
    timestamps=np.concatenate([b['batch_absolute_timestamp'].numpy().reshape(-1) for b in loaders[2]])
    np.savez(out/'predictions.npz',prediction=predicted[valid],actual=actual[valid],
        variable_id=trainer.last_eval_var_idx[valid],window_id=trainer.last_eval_window_id[valid],timestamp=timestamps[valid])
    provenance=json.loads((Path(__file__).parents[1]/f'{args.method}_irregular/vendor/PROVENANCE.json').read_text())
    manifest=dict(method=args.method,variant=PROFILES[args.method]['variant'] if args.mode=='FullExo' else args.method+'-Y-only',
        mode=args.mode,arguments=vars(args),source_config=config,model_config=model.model_config,
        upstream=provenance,split=global_args['exen_processed_summary'],
        best_validation=best,test=result,training_history=trainer.evaluation_history[:-1],
        evaluation_scaler=evaluation.manifest(),input_scaler='train-only MinMax',
        history_grid_seconds=(model.history_grid.cpu().numpy()*p['duration_seconds']).tolist(),
        future_grid_seconds=(model.future_grid.cpu().numpy()*p['duration_seconds']).tolist(),
        known_future_exogenous_var_ids=sorted(loaders[0].dataset.known_future_exogenous_var_ids),
        target_ids=p['target_ids'],exogenous_ids=p['exogenous_ids'],
        parameter_count=sum(v.numel() for v in model.parameters()),objective=args.loss,
        optimizer='Adam',learning_rate_schedule=lradj,
        checkpoint_selection=f'validation query {args.loss} in input units; equal batch weight',
        metric_aggregation='equal window weight; real valid Y queries only',
        forecast_inputs='Y_history_X_history_and_declared_known_future_X' if args.mode=='FullExo' else 'Y_history_only',
        time_mapping='history step H/seq_len; future origin + k*step, k=0..ceil(F/step); linear output interpolation')
    (out/'metrics.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),test=result),indent=2))
    return manifest


def main(method):
    run(make_parser(method).parse_args())
