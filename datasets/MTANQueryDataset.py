"""mTAN dense ragged histories; reuse existing task/query/MLP conventions."""

import torch

from .QueryFusionDataset import QueryFusionDataset


class MTANQueryDataset(QueryFusionDataset):
    def long_collate_fn(self, batch):
        out = super().long_collate_fn(batch)
        histories = []
        for b in range(len(batch)):
            valid = out["batch_pad_mask"][b].bool()
            role = out["batch_node_type"][b]
            take = valid & (
                (role == 1) | ((role == 0) & (self.backbone_history == "yx"))
            )
            indices = torch.where(take)[0]
            # Equal-time distinct variables share a row. Repeated observations
            # of the same variable at the same time get distinct rows, not sums.
            counts, nodes = {}, []
            for ix in indices.tolist():
                time = float(out["batch_timestamp"][b, ix])
                variable = self.local_id[int(out["batch_var_idx"][b, ix])]
                key = (time, variable)
                occurrence = counts.get(key, 0)
                counts[key] = occurrence + 1
                nodes.append(((time, occurrence), variable, ix))
            keys = sorted({node[0] for node in nodes})
            slots = {key: i for i, key in enumerate(keys)}
            n = max(1, len(keys))
            values = torch.zeros(n, len(self.backbone_ids))
            times = torch.zeros(n)
            mask = torch.zeros_like(values)
            for key, column, ix in nodes:
                row = slots[key]
                values[row, column] = out["batch_value"][b, ix]
                times[row] = key[0]
                mask[row, column] = 1
            histories.append((values, times, mask))
        length = max(len(v) for v, _, _ in histories)
        values = torch.zeros(len(batch), length, len(self.backbone_ids))
        times = torch.zeros(len(batch), length)
        mask = torch.zeros_like(values)
        for b, (v, t, m) in enumerate(histories):
            values[b, : len(v)] = v
            times[b, : len(t)] = t
            mask[b, : len(m)] = m
        # The parent's patches are only a collate helper; mTAN never sees them.
        for key in ("values", "times", "mask"):
            out.pop("batch_backbone_" + key)
        out.update(
            batch_backbone_mtan_values=values,
            batch_backbone_mtan_times=times,
            batch_backbone_mtan_mask=mask,
        )
        return out
