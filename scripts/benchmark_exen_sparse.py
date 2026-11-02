"""Benchmark one actual processed-data window, without training an epoch."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from data_processors import ExEnProcessedProcessor
from datasets import ExEnProcessedDataset
from utils.scaler import ExEnProcessedMinMaxScaler
from models import ExEn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/ExEn/processed/DIST.yaml')
    parser.add_argument('--backend', choices=['dense', 'sparse_csr'], required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', default='results/sparse_benchmark')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(123)
    config = yaml.safe_load(Path(args.config).read_text())
    g = config['global_config']; g['device'] = args.device
    data = ExEnProcessedProcessor(g, **config['data_config']['data_processor_params']).process()
    scaler = ExEnProcessedMinMaxScaler(); scaler.fit(data['train_data'])
    dataset = ExEnProcessedDataset(scaler.transform(data['train_data']), type='train', global_args=g,
                                   **config['data_config']['dataset_params'])
    batch = {k: v.to(args.device) for k, v in dataset.collate_fn([dataset[0]]).items()}
    options = dict(config['model_config']['model_params'], relation_backend=args.backend)
    model = ExEn(global_args=g, **options).to(args.device)
    cuda = torch.device(args.device).type == 'cuda'
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prediction = model(**batch)
    query = batch['batch_pred_mask'].bool()
    loss = (prediction.squeeze(-1)[query] - batch['batch_value'][query]).square().mean()
    loss.backward()
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(backend=args.backend, device=args.device, threads=args.threads, seed=123,
        torch_version=torch.__version__, nodes=int(batch['batch_pad_mask'].sum()),
        hidden_dim=options['hid_dim'], temporal_layers=options['n_layer'],
        cross_layers=options['n_cross_layer'], receiver_chunk_size=options.get('receiver_chunk_size'),
        forward_backward_seconds=elapsed,
        peak_process_rss_bytes=peak_rss if sys.platform == 'darwin' else peak_rss * 1024,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
        finite=bool(torch.isfinite(loss) and all(torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.grad is not None)),
        scope='One first training window; timing excludes loading and scaler fitting; not epoch throughput')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / f'{args.backend}.json').write_text(json.dumps(result, indent=2))
    torch.save(dict(prediction=prediction.detach().cpu(),
                    gradients={k: v.grad.cpu() for k,v in model.named_parameters() if v.grad is not None}),
               output / f'{args.backend}.pt')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
