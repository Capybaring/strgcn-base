"""Exact legal-edge relations in streamed CSR form; never construct [L,L].

Each sorted source list and row degree encodes a CSR row's causal prefix.
Columns/values are materialized for one receiver block at a time, then released.
"""
from dataclasses import dataclass
import math
import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class RelationGroup:
    batch: int
    relation: int  # 0=X time, 1=Y time, 2=X->Y
    sources: torch.Tensor
    receivers: torch.Tensor
    degrees: torch.Tensor


@torch.no_grad()
def build_sparse_relations(timestamp, var_idx, node_type, valid):
    groups = []
    for batch in range(timestamp.shape[0]):
        t, v, role, keep = timestamp[batch], var_idx[batch], node_type[batch], valid[batch]
        exo = (role == 0) | (role == 2)
        known = keep & (role != 3)
        candidates = []
        for variable in torch.unique(v[keep]).tolist():
            dst = torch.where(keep & (v == variable))[0]
            src = torch.where(known & (v == variable))[0]
            if len(dst):
                relation = 0 if bool(exo[dst[0]]) else 1
                candidates.append((relation, src, dst))
        candidates.append((2, torch.where(known & exo)[0], torch.where(keep & ~exo)[0]))
        for relation, src, dst in candidates:
            if not len(src) or not len(dst):
                continue
            src = src[torch.argsort(t[src], stable=True)]
            degree = torch.searchsorted(t[src].contiguous(), t[dst].contiguous(), right=True)
            active = degree > 0
            dst, degree = dst[active], degree[active]
            if len(dst):
                src = src[:int(degree.max())]
                groups.append(RelationGroup(batch, relation, src, dst, degree))
    return groups


def csr_block(p_receiver, p_source, source_value, u_receiver, v_source,
              degrees, time_scale, var_scale, cross):
    """Sample dot products only at legal CSR entries, softmax by CSR row."""
    crow = torch.cat((degrees.new_zeros(1), degrees.cumsum(0)))
    edge_count = int(crow[-1])
    # Every row's neighbors are source columns 0..degree-1 (sorted by time).
    col = torch.arange(edge_count, device=degrees.device) - torch.repeat_interleave(
        crow[:-1], degrees, output_size=edge_count)
    pattern = torch.sparse_csr_tensor(crow, col, p_receiver.new_ones(edge_count),
        size=(len(degrees), len(p_source)), check_invariants=False)
    logits = torch.sparse.sampled_addmm(pattern, p_receiver, p_source.T, beta=0).values() * time_scale
    if cross:
        support = torch.sparse.sampled_addmm(pattern, u_receiver, v_source.T, beta=0).values()
        logits = logits + var_scale * support / math.sqrt(u_receiver.shape[-1])
    maximum = torch.segment_reduce(logits, 'max', lengths=degrees)
    exps = (logits - maximum.repeat_interleave(degrees, output_size=edge_count)).exp()
    denominator = torch.segment_reduce(exps, 'sum', lengths=degrees)
    weights = exps / denominator.repeat_interleave(degrees, output_size=edge_count)
    adjacency = torch.sparse_csr_tensor(crow, col, weights, size=pattern.shape, check_invariants=False)
    return torch.sparse.mm(adjacency, source_value)


def sparse_messages(layer, h, p, var_idx, groups, chunk_size):
    """Same low-rank source transform as low_rank_graph_conv, computed once."""
    if h.device.type not in ('cpu', 'cuda'):
        raise ValueError('sparse_csr relations require CPU or CUDA; use dense on other devices')
    transformed = [(h * transform(p)).sum(-1, keepdim=True) * embedding(var_idx)
                   for transform, embedding in zip(layer.weight_T, layer.weight_S)]
    if layer.cross:
        all_u, all_v = layer.adj_U(var_idx), layer.adj_V(var_idx)
    messages = [torch.zeros_like(row) for row in h]
    receiver = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
    for group in groups:
        if (group.relation == 2) != layer.cross:
            continue
        b, src, dst = group.batch, group.sources, group.receivers
        k = 0 if layer.cross else group.relation
        p_source = p[b, src]
        value = transformed[k][b, src]
        v_source = all_v[b, src] if layer.cross else p_source[:, :0]
        for start in range(0, len(dst), chunk_size):
            nodes = dst[start:start+chunk_size]
            degrees = group.degrees[start:start+chunk_size]
            p_receiver = p[b, nodes]
            u_receiver = all_u[b, nodes] if layer.cross else p_receiver[:, :0]
            var_scale = layer.var_scale if layer.cross else layer.time_scale[k].new_zeros(())
            def aggregate(pi, pj, hv, ui, vj, degree=degrees, index=k, scale=var_scale):
                return csr_block(pi, pj, hv, ui, vj, degree, layer.time_scale[index], scale, layer.cross)
            args = (p_receiver, p_source, value, u_receiver, v_source)
            if layer.training and torch.is_grad_enabled():
                result = checkpoint(aggregate, *args, use_reentrant=False)
            else:
                result = aggregate(*args)
            messages[b] = messages[b].index_add(0, nodes, result)
        receiver[b, dst] = True
    # Match dense zero gradients for absent relation types, including optimizers
    # that distinguish a missing gradient from an explicitly zero gradient.
    zero = sum(value.sum() * 0 for value in transformed) + layer.time_scale.sum() * 0
    if layer.cross:
        zero = zero + (all_u.sum() + all_v.sum() + layer.var_scale) * 0
    return torch.stack(messages) + zero, receiver
