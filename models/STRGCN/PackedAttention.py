"""CPU-built, size-bucketed legal relations for batched observation attention."""
from dataclasses import dataclass
import torch
from .SparseRelations import build_sparse_relations, build_within_variable_relations


@dataclass
class AttentionBucket:
    relation: int
    source: torch.Tensor
    receiver: torch.Tensor
    degrees: torch.Tensor
    positions: torch.Tensor
    destinations: torch.Tensor

    def to(self, device, non_blocking=False):
        return AttentionBucket(self.relation, *(x.to(device, non_blocking=non_blocking)
            for x in (self.source, self.receiver, self.degrees, self.positions, self.destinations)))


@dataclass
class PackedAttentionPlan:
    buckets: list
    within_relations: tuple = ()
    windows: list | None = None

    def to(self, device, non_blocking=False):
        return PackedAttentionPlan([b.to(device, non_blocking) for b in self.buckets], self.within_relations,
            None if self.windows is None else [w.to(device, non_blocking) for w in self.windows])


def pack_attention_plan(timestamp, var_idx, node_type, valid, receiver_chunk=512,
                        max_score_elements=1048576, within_relations=(), groups=None):
    """Called in CPU collate: no per-variable host reads during CUDA forward.

    Power-of-two padding groups similarly sized relations. Budget is score slots
    per head; an individually oversized group is receiver-chunked first.
    """
    if timestamp.device.type != 'cpu':
        raise ValueError('Build packed relation metadata on CPU before transfer')
    if receiver_chunk < 1 or max_score_elements < 1:
        raise ValueError('Attention budgets must be positive')
    width = timestamp.shape[1]
    if groups is None:
        groups = build_sparse_relations(timestamp, var_idx, node_type, valid)
        if within_relations:
            groups += build_within_variable_relations(timestamp, var_idx, node_type, valid, within_relations)
    by_size = {}
    def rounded(n): return 1 << (n - 1).bit_length()
    for group in groups:
        for offset in range(0, len(group.receivers), receiver_chunk):
            dst = group.receivers[offset:offset + receiver_chunk]
            degree = group.degrees[offset:offset + receiver_chunk]
            src = group.sources[:int(degree.max())]
            key = (group.relation, rounded(len(dst)), rounded(len(src)))
            by_size.setdefault(key, []).append((group.batch * width + src,
                                                group.batch * width + dst, degree))
    buckets = []
    for (relation, qlen, slen), entries in sorted(by_size.items()):
        batch_limit = max(1, max_score_elements // (qlen * slen))
        for begin in range(0, len(entries), batch_limit):
            subset = entries[begin:begin + batch_limit]
            src = torch.zeros(len(subset), slen, dtype=torch.long)
            dst = torch.zeros(len(subset), qlen, dtype=torch.long)
            deg = torch.zeros_like(dst)
            pos, destinations = [], []
            for i, (s, d, n) in enumerate(subset):
                src[i, :len(s)] = s
                dst[i, :len(d)] = d
                deg[i, :len(d)] = n
                pos.append(torch.arange(len(d)) + i * qlen)
                destinations.append(d)
            buckets.append(AttentionBucket(relation, src, dst, deg,
                torch.cat(pos), torch.cat(destinations)))
    return PackedAttentionPlan(buckets, tuple(within_relations))
