"""Matched-batch forward/backward timing, no optimizer updates or accuracy claim."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_processors import ExEnProcessedProcessor
from datasets import ExEnProcessedDataset
from models import ExEnAligned,ExEnGraphTransformer
from utils.scaler import ExEnProcessedMinMaxScaler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='NP')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--attention-dtype',choices=['float32','auto','bfloat16','float16'],default='float32')
    parser.add_argument('--no-exogenous',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if min(args.batch_size,args.steps,args.threads)<1 or args.warmup<0:parser.error('Invalid batch size, steps, threads or warmup')
    torch.set_num_threads(args.threads)
    config=ROOT/'config/ExEnGraphTransformer/processed'/f'{args.dataset}.yaml'
    c=yaml.safe_load(config.read_text());g=dict(c['global_config'],device=args.device)
    g['exen_data_path']=str(ROOT/g['exen_data_path'])
    data=ExEnProcessedProcessor(g,**c['data_config']['data_processor_params']).process()
    scale=ExEnProcessedMinMaxScaler();scale.fit(data['train_data'])
    ds=ExEnProcessedDataset(scale.transform(data['train_data']),type='train',global_args=g,
        use_exogenous=not args.no_exogenous,**c['data_config']['dataset_params'])
    batch=ds.collate_fn([ds[i] for i in range(min(args.batch_size,len(ds)))])
    batch={k:v.to(args.device) for k,v in batch.items()};query=batch['batch_pred_mask'].bool()
    report=dict(scope='Same first actual training batch, no optimizer steps; excludes data loading, scaler fitting and warmup. Not full-epoch throughput or accuracy.',
                dataset=args.dataset,device=args.device,torch_version=torch.__version__,threads=args.threads,
                batch_size=batch['batch_value'].shape[0],padded_nodes=batch['batch_value'].shape[1],
                actual_nodes=int(batch['batch_pad_mask'].sum()),query_count=int(query.sum()),
                use_exogenous=not args.no_exogenous,attention_dtype=args.attention_dtype,results=[])
    for name,cls in [('ExEnAligned',ExEnAligned),('ExEnGraphTransformer',ExEnGraphTransformer)]:
        cfg=yaml.safe_load((ROOT/f'config/{name}/processed/{args.dataset}.yaml').read_text())
        params=dict(cfg['model_config']['model_params'])
        if name=='ExEnGraphTransformer':params['attention_dtype']=args.attention_dtype
        torch.manual_seed(c['train_config']['random_seed'])
        gc.collect()
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
        m=cls(global_args=g,**params).to(args.device).train()
        def step():
            m.zero_grad(set_to_none=True)
            pred=m(**batch).squeeze(-1)
            loss=(pred[query]-batch['batch_value'][query]).square().mean()
            loss.backward()
            return float(loss.detach())
        for _ in range(args.warmup):step()
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device);torch.cuda.reset_peak_memory_stats(args.device)
        times=[]
        for _ in range(args.steps):
            start=time.perf_counter();loss=step()
            if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
            times.append(time.perf_counter()-start)
        finite=all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)
        if not finite:raise RuntimeError(f'Nonfinite gradients in {name}')
        result=dict(model=name,parameters=sum(p.numel() for p in m.parameters()),
                    receiver_chunk_size=params.get('receiver_chunk_size'),
                    median_forward_backward_seconds=statistics.median(times),measurements_seconds=times,
                    finite_gradients=finite)
        if args.device.startswith('cuda'):result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
        report['results'].append(result);del m;gc.collect()
    report['graph_transformer_speedup']=report['results'][0]['median_forward_backward_seconds']/report['results'][1]['median_forward_backward_seconds']
    print(json.dumps(report,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
