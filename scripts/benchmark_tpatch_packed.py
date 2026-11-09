"""Compare loop/checkpoint and packed attention on identical real task samples."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from data_processors import TPatchRecordProcessor
from datasets import TPatchRecordDataset
from models import ExEnGraphTransformer
from utils.scaler import TPatchInputScaler
from utils.tpatch_evaluation import variable_macro_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['activity','mimic','physionet','ushcn'], default='activity')
    p.add_argument('--data-root', default='data/tpatch-data')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', default='results/tpatch_packed_benchmark.json')
    args = p.parse_args()
    if args.repeats < 1 or (args.batch_size is not None and args.batch_size < 1):
        p.error('repeats and batch-size must be positive')
    torch.set_num_threads(1); torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    c = yaml.safe_load((root/f'config/ExEnGraphTransformer/tpatch/{args.dataset}.yaml').read_text())
    size = args.batch_size or c['train_config']['batch_size']
    g = {'device':args.device}
    data = TPatchRecordProcessor(g,args.dataset,data_root=args.data_root).process()
    scale = TPatchInputScaler();scale.fit(data['seen_data'])
    packet = scale.transform(data['train_data'])
    ds = TPatchRecordDataset(packet,type='train',global_args=g)
    # Stable evenly spread tasks; no selecting convenient low-density records.
    indices = np.linspace(0,len(ds)-1,min(size,len(ds)),dtype=int)
    sample = [ds[int(i)] for i in indices]
    cpu = ds.collate_fn(sample)
    from models.STRGCN.PackedAttention import pack_attention_plan
    start=time.perf_counter()
    plan=pack_attention_plan(cpu['batch_timestamp'],cpu['batch_var_idx'],cpu['batch_node_type'],cpu['batch_pad_mask'].bool())
    plan_ms=(time.perf_counter()-start)*1000
    b={k:v.to(args.device) for k,v in cpu.items()}
    plan=plan.to(args.device)
    options=c['model_config']['model_params']
    baseline=ExEnGraphTransformer(global_args=g,**dict(options,checkpoint_attention=True)).to(args.device)
    packed=ExEnGraphTransformer(global_args=g,**dict(options,checkpoint_attention=False)).to(args.device)
    packed.load_state_dict(baseline.state_dict())
    cuda=torch.device(args.device).type=='cuda'
    def sync():
        if cuda:torch.cuda.synchronize()
    def measure(model,packed_plan):
        model.train()
        def step():
            model.zero_grad(set_to_none=True)
            pred=model(**b,**({'batch_attention_plan':packed_plan} if packed_plan is not None else {})).squeeze(-1)
            loss=variable_macro_loss(pred,b['batch_value'],b['batch_pred_mask'].bool()&b['batch_pad_mask'].bool(),b['batch_var_idx'],g['num_of_vals'])
            loss.backward()
        step();sync()
        if cuda:torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(args.repeats):
            sync();t=time.perf_counter();step();sync();times.append((time.perf_counter()-t)*1000)
        return dict(median_ms=float(np.median(times)),trials_ms=times,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None)
    old=measure(baseline,None)
    baseline.zero_grad(set_to_none=True)
    new=measure(packed,plan)
    baseline.eval();packed.eval()
    with torch.no_grad():
        expected=baseline(**b);actual=packed(**b,batch_attention_plan=plan)
    error=float((expected-actual).abs().max().cpu())
    result=dict(dataset=args.dataset,device=args.device,torch=torch.__version__,batch_size=len(sample),
        nodes=int(cpu['batch_pad_mask'].sum()),cpu_plan_build_ms=plan_ms,
        packed_sdpa_calls_forward=4*sum(x.relation!=2 for x in plan.buckets)+3*sum(x.relation==2 for x in plan.buckets),
        old_loop_checkpoint=old,new_packed_no_checkpoint=new,
        model_step_speedup=old['median_ms']/new['median_ms'],max_eval_prediction_abs_difference=error,
        scope='same observations and 4+3-layer weights, model forward/backward only; excludes optimizer/data; separate real GPU benchmarking required')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
