"""CPU preparation of historical patches and pointwise DAG future-X inputs."""
import math
import torch
from .TPatchRecordDataset import TPatchRecordDataset


class TPatchFusionDataset(TPatchRecordDataset):
    def __init__(self, *args, patch_size, **kwargs):
        if not math.isfinite(patch_size) or patch_size <= 0:
            raise ValueError('patch_size must be positive and finite in native time units')
        kwargs['pack_attention'] = False
        super().__init__(*args, **kwargs)
        self.patch_size = float(patch_size)
        self.backbone_ids = list(self.packet['target_ids']) + sorted(self.exogenous_var_ids)
        self.local_id = {v: j for j, v in enumerate(self.backbone_ids)}
        history = self.packet['history_native']
        self.npatch = math.ceil(history / self.patch_size)
        if self.npatch < 2:
            raise ValueError('Patch baselines require at least two history patches')
        self.global_args.update(fusion_target_ids=list(self.packet['target_ids']),
            fusion_exogenous_ids=sorted(self.exogenous_var_ids),
            fusion_backbone_ids=self.backbone_ids, fusion_num_patches=self.npatch,
            fusion_future_fill='last observed X at or before query; zero without prior value',
            fusion_inputs='predicted Y vector + future X vector only; no mask/delta/static branch')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        start = self.packet['windows'][index][4]
        # Parent long collate accepts (sample, metadata); aggregation is disabled.
        origin = start / self.packet['time_scale'] if self.packet['preserve_record_time'] else 0.
        return sample, torch.tensor(origin, dtype=torch.float64)

    def long_collate_fn(self, batch):
        out = super().long_collate_fn(batch)
        B, width = out['batch_value'].shape
        N, M = len(self.backbone_ids), self.npatch
        patch_span = self.patch_size / self.packet['time_scale']
        groups, queries, future_vectors, locations = [], [], [], []
        local_map = torch.full((self.global_args['num_of_vals'],), -1, dtype=torch.long)
        local_map[self.backbone_ids] = torch.arange(N)
        max_length = 1
        for b, (_, origin) in enumerate(batch):
            origin = float(origin)
            time = out['batch_timestamp'][b]
            value = out['batch_value'][b]
            ids = out['batch_var_idx'][b]
            valid = out['batch_pad_mask'][b].bool()
            role = out['batch_node_type'][b]
            historical = valid & ((role == 0) | (role == 1))
            qmask = valid & out['batch_pred_mask'][b].bool()
            q = torch.unique(time[qmask], sorted=True)
            queries.append(q)
            locations.append((torch.where(qmask)[0], torch.searchsorted(q, time[qmask])))
            # Group all history nodes in one pass rather than B*M*N masks.
            take = historical & (local_map[ids] >= 0)
            t, v, j = time[take], value[take], local_map[ids[take]]
            edges = time.new_tensor([origin + k * patch_span for k in range(M + 1)])
            patch = torch.searchsorted(edges, t.contiguous(), right=True) - 1
            inside = (patch >= 0) & (patch < M)
            patch, t, v, j = patch[inside], t[inside], v[inside], j[inside]
            key = patch * N + j
            order = torch.argsort(key, stable=True)
            count = torch.bincount(key, minlength=M * N)
            start = count.cumsum(0) - count
            within = torch.arange(len(order)) - torch.repeat_interleave(start, count)
            max_length = max(max_length, int(count.max()))
            groups.append((patch[order], within, j[order], v[order], t[order]))
            xq = value.new_zeros(len(q), len(self.exogenous_var_ids))
            for column, vid in enumerate(sorted(self.exogenous_var_ids)):
                if not self.use_exogenous or vid not in self.known_future_exogenous_var_ids:
                    continue
                take = valid & (ids == vid) & ((role == 0) | (role == 2))
                t, v = time[take], value[take]
                if not len(t):
                    continue
                order = torch.argsort(t, stable=True); t, v = t[order], v[order]
                previous = torch.searchsorted(t.contiguous(), q.contiguous(), right=True) - 1
                available = previous >= 0
                xq[available, column] = v[previous[available]]
            future_vectors.append(xq)
        Q = max(map(len, queries))
        values = torch.zeros(B, M, max_length, N)
        times, mask = torch.zeros_like(values), torch.zeros_like(values)
        qt = torch.zeros(B, Q)
        fx = torch.zeros(B, Q, len(self.exogenous_var_ids))
        query_slot = torch.zeros(B, width, dtype=torch.long)
        for b, (patch, within, variable, v, t) in enumerate(groups):
            values[b, patch, within, variable] = v
            times[b, patch, within, variable] = t
            mask[b, patch, within, variable] = 1
            n = len(queries[b]); qt[b, :n] = queries[b]; fx[b, :n] = future_vectors[b]
            indices, slots = locations[b]; query_slot[b, indices] = slots
        out.update(batch_backbone_values=values, batch_backbone_times=times,
                   batch_backbone_mask=mask, batch_backbone_queries=qt,
                   batch_backbone_future_x=fx, batch_backbone_query_slot=query_slot)
        return out
