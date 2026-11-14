"""CPU-built, size-bucketed legal relations for batched observation attention."""
from dataclasses import dataclass, fields, is_dataclass
import numpy as np
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
        return transfer_attention_plan(self, device, non_blocking)



def transfer_attention_plan(plan, device, non_blocking=False):
    """Transfer the complete metadata tree in one contiguous slab per dtype.

    Thousands of independent small H2D copies serialize pageable transfers even
    with non_blocking=True. Shared tensors are deduplicated; views own the moved
    slab and remain valid after the CPU plan has been released.
    """
    tensors = {}
    def collect(item):
        if isinstance(item, torch.Tensor):
            tensors.setdefault(id(item), item)
        elif is_dataclass(item):
            for field in fields(item): collect(getattr(item, field.name))
        elif isinstance(item, (list, tuple)):
            for value in item: collect(value)
    collect(plan)
    groups = {}
    for identity, tensor in tensors.items():
        groups.setdefault((tensor.dtype, tensor.device), []).append((identity, tensor))
    moved = {}
    for entries in groups.values():
        slab = torch.cat([tensor.reshape(-1) for _, tensor in entries])
        slab = slab.to(device, non_blocking=non_blocking)
        offset = 0
        for identity, tensor in entries:
            size = tensor.numel()
            moved[identity] = slab[offset:offset + size].view(tensor.shape)
            offset += size
    def rebuild(item):
        if isinstance(item, torch.Tensor): return moved[id(item)]
        if is_dataclass(item):
            return type(item)(**{field.name: rebuild(getattr(item, field.name)) for field in fields(item)})
        if isinstance(item, list): return [rebuild(value) for value in item]
        if isinstance(item, tuple): return tuple(rebuild(value) for value in item)
        return item
    return rebuild(plan)


def pack_attention_entries(entries, receiver_chunk=512, max_score_elements=1048576,
                           within_relations=()):
    """Bulk NumPy packing of CPU (relation,offset,sources,receivers,degrees)."""
    if receiver_chunk < 1 or max_score_elements < 1:
        raise ValueError('Attention budgets must be positive')
    by_size = {}
    def rounded(n): return 1 << (n - 1).bit_length()
    for relation, batch_offset, source, receivers, degrees in entries:
        count = receivers.shape[0]
        for offset in range(0, count, receiver_chunk):
            dst = receivers[offset:offset + receiver_chunk]
            degree = degrees[offset:offset + receiver_chunk]
            nsource = int(degree.max())
            key = (relation, rounded(dst.shape[0]), rounded(nsource))
            by_size.setdefault(key, []).append((batch_offset, source[:nsource], dst, degree))
    buckets = []
    for (relation, qlen, slen), entries in sorted(by_size.items()):
        batch_limit = max(1, max_score_elements // (qlen * slen))
        for begin in range(0, len(entries), batch_limit):
            subset = entries[begin:begin + batch_limit]
            src = np.zeros((len(subset), slen), dtype=np.int64)
            dst = np.zeros((len(subset), qlen), dtype=np.int64)
            deg = np.zeros_like(dst)
            sizes = np.empty(len(subset), dtype=np.int64)
            for i, (base, source, receiver, degree) in enumerate(subset):
                size = receiver.shape[0]
                src[i, :source.shape[0]] = source + base
                dst[i, :size] = receiver + base
                deg[i, :size] = degree
                sizes[i] = size
            selected = np.arange(qlen)[None, :] < sizes[:, None]
            positions = np.flatnonzero(selected)
            destinations = dst[selected]
            buckets.append(AttentionBucket(relation, *(torch.from_numpy(array) for array in
                (src, dst, deg, positions, destinations))))
    return PackedAttentionPlan(buckets, tuple(within_relations))


def pack_attention_plan(timestamp, var_idx, node_type, valid, receiver_chunk=512,
                        max_score_elements=1048576, within_relations=(), groups=None):
    """Build bounded power-of-two buckets on CPU before device transfer."""
    if timestamp.device.type != 'cpu':
        raise ValueError('Build packed relation metadata on CPU before transfer')
    if groups is None:
        groups = build_sparse_relations(timestamp, var_idx, node_type, valid)
        if within_relations:
            groups += build_within_variable_relations(timestamp, var_idx, node_type, valid, within_relations)
    width = timestamp.shape[1]
    entries = ((g.relation, g.batch * width, g.sources.numpy(), g.receivers.numpy(), g.degrees.numpy())
               for g in groups)
    return pack_attention_entries(entries, receiver_chunk, max_score_elements, within_relations)
