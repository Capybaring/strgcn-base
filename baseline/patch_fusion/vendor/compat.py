"""Exact gather/message/sum fallback for this vendored Hi-Patch layer only.

Use the original PyG / torch_scatter packages when installed. This fallback
implements only the add-aggregation API used by Hi-Patch, not general PyG.
"""
import torch
from torch import nn


class MessagePassing(nn.Module):
    def __init__(self, aggr='add', **kwargs):
        super().__init__()
        if aggr != 'add' or kwargs:
            raise ValueError('Hi-Patch fallback supports add aggregation only')

    def propagate(self, edge_index, x, edges_temporal, edge_same_time_diff_var,
                  edge_diff_time_same_var, edge_diff_time_diff_var, n_layer, residual):
        source, target = edge_index
        messages = self.message(x[source], x[target], target, edges_temporal,
            edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer)
        aggregate = torch.zeros_like(x).index_add(0, target, messages)
        return self.update(aggregate, residual)


def scatter(src, index, dim=0, dim_size=None, reduce='sum'):
    if dim != 0 or reduce != 'sum':
        raise ValueError('Hi-Patch fallback supports scatter sum on dim=0 only')
    n = maybe_num_nodes(index) if dim_size is None else dim_size
    return src.new_zeros((n,) + src.shape[1:]).index_add(0, index, src)


def maybe_num_nodes(index):
    return int(index.max()) + 1 if index.numel() else 0
