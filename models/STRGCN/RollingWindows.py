"""Chronological observation graphs with differentiable, ordered window memory.

Plans contain metadata only, are built on CPU, and preserve empty time intervals.
No labels, query states, predicted Y, or detached learned states enter memory.
All times and window_size use the dataset's existing normalized time coordinate.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .PackedAttention import AttentionBucket, PackedAttentionPlan, pack_attention_plan
from .SparseRelations import RelationGroup, build_sparse_relations, build_within_variable_relations
from .STRGCN_Layer import masked_softmax


@dataclass
class WindowLayout:
    current: torch.Tensor
    batch: torch.Tensor
    known: torch.Tensor  # local positions of current observations
    summary: torch.Tensor  # padded local positions, [B,max_current_observations]
    summary_valid: torch.Tensor
    plan: PackedAttentionPlan  # receiver/current and source/cache coordinates

    def to(self, device, non_blocking=False):
        return WindowLayout(*(x.to(device, non_blocking=non_blocking) for x in
            (self.current, self.batch, self.known, self.summary, self.summary_valid)),
            self.plan.to(device, non_blocking))


def _compact_layout(plan, receivers, node_type, history):
    """Compile indices on CPU; no CUDA nonzero or boolean-gather synchronization."""
    current = torch.where(receivers.flatten())[0]
    width = receivers.shape[1]
    batches = current // width
    known = torch.where(node_type.flatten()[current] != 3)[0]
    sources = torch.cat((history, current[known]))
    qmap = torch.zeros(receivers.numel(), dtype=torch.long)
    smap = torch.zeros_like(qmap)
    qmap[current] = torch.arange(len(current))
    smap[sources] = torch.arange(len(sources))
    buckets = [AttentionBucket(b.relation, smap[b.source], qmap[b.receiver],
        b.degrees, b.positions, qmap[b.destinations]) for b in plan.buckets]
    rows = [known[batches[known] == b] for b in range(len(receivers))]
    length = max(map(len, rows), default=0)
    summary = torch.zeros(len(rows), length, dtype=torch.long)
    keep = torch.zeros_like(summary, dtype=torch.bool)
    for b, row in enumerate(rows):
        summary[b, :len(row)] = row
        keep[b, :len(row)] = True
    layout = WindowLayout(current, batches, known, summary, keep,
        PackedAttentionPlan(buckets, plan.within_relations))
    return layout, sources


@dataclass
class WindowStep:
    plan: PackedAttentionPlan
    receivers: torch.Tensor
    bounds: torch.Tensor  # [B,2], absolute normalized start/end
    present: torch.Tensor  # padded records must not add fictitious window tokens
    layout: WindowLayout | None = None

    def to(self, device, non_blocking=False):
        return WindowStep(self.plan.to(device, non_blocking),
            self.receivers.to(device, non_blocking=non_blocking),
            self.bounds.to(device, non_blocking=non_blocking),
            self.present.to(device, non_blocking=non_blocking),
            None if self.layout is None else self.layout.to(device, non_blocking))


def _intervals(start, forecast, end, size):
    intervals = []
    # Forecast origin is always a boundary, even if the history is not divisible.
    for left, right in ((start, forecast), (forecast, end)):
        if right <= left:
            continue
        count = max(1, math.ceil((right - left) / size - 1e-6))
        for k in range(count):
            intervals.append((left + k * size, right if k == count - 1 else min(left + (k + 1) * size, right)))
    return intervals


@torch.no_grad()
def pack_rolling_attention_plan(timestamp, var_idx, node_type, valid, window_size,
                                bounds=None, within_relations=(), receiver_chunk=512,
                                max_score_elements=1048576):
    """bounds=[start, forecast_origin, end] per record, independent of sampling.

    Exact bounds are supplied by datasets. Fallback bounds support direct model
    calls but cannot reconstruct unobserved leading/trailing task intervals.
    """
    if timestamp.device.type != 'cpu':
        raise ValueError('Build rolling metadata on CPU')
    if not math.isfinite(window_size) or window_size <= 0:
        raise ValueError('window_size must be positive and finite')
    valid = valid.bool()
    if bounds is None:
        rows = []
        for time, role, keep in zip(timestamp, node_type, valid):
            if not keep.any():
                rows.append([0., 0., 0.]); continue
            start = float(time[keep].min())
            # Include a possible observation exactly on the last timestamp.
            end = float(time[keep].max()) + max(window_size * 1e-5, 1e-7)
            future = keep & (role >= 2)
            origin = float(time[future].min()) if future.any() else end
            rows.append([start, origin, end])
        bounds = timestamp.new_tensor(rows)
    if bounds.shape != (timestamp.shape[0], 3) or not torch.isfinite(bounds).all():
        raise ValueError('Window bounds must be finite [B,3] metadata')
    schedules = []
    for b, (start, origin, end) in enumerate(bounds.tolist()):
        if not start <= origin <= end:
            raise ValueError('Expected start <= forecast_origin <= end')
        if ((timestamp[b, valid[b]] < start) | (timestamp[b, valid[b]] > end)).any():
            raise ValueError('Valid nodes must lie inside task bounds [start,end]')
        future = valid[b] & (node_type[b] >= 2)
        history = valid[b] & ~future
        if (timestamp[b, history] >= origin).any() or (timestamp[b, future] < origin).any():
            raise ValueError('Node roles disagree with forecast boundary')
        schedules.append(_intervals(start, origin, end, window_size))
    groups = build_sparse_relations(timestamp, var_idx, node_type, valid)
    groups += build_within_variable_relations(timestamp, var_idx, node_type, valid, within_relations)
    metadata = []
    coverage = torch.zeros_like(timestamp, dtype=torch.long)
    window_ids = torch.full_like(node_type, -1, dtype=torch.long)
    for k in range(max(map(len, schedules), default=0)):
        receivers = torch.zeros_like(valid)
        edges = timestamp.new_zeros((len(schedules), 2))
        present = torch.zeros(len(schedules), dtype=torch.bool)
        for b, schedule in enumerate(schedules):
            if k < len(schedule):
                left, right = schedule[k]
                edges[b] = timestamp.new_tensor([left, right])
                upper = timestamp[b] <= right if k == len(schedule) - 1 else timestamp[b] < right
                receivers[b] = valid[b] & (timestamp[b] >= left) & upper
                present[b] = True
        coverage += receivers.long()
        window_ids[receivers] = k
        metadata.append((receivers, edges, present))
    if not torch.equal(coverage, valid.long()):
        raise ValueError('Every valid node must belong to exactly one chronological window')
    # Split each source group once, rather than rescanning every group K times.
    # Python integer lists avoid many small CPU tensor reductions/indexing calls.
    local_groups = [[] for _ in metadata]
    for group in groups:
        by_window = {}
        for k, dst, degree in zip(window_ids[group.batch, group.receivers].tolist(),
                                  group.receivers.tolist(), group.degrees.tolist()):
            nodes, degrees = by_window.setdefault(k, ([], []))
            nodes.append(dst); degrees.append(degree)
        for k, (nodes, degrees) in by_window.items():
            local_groups[k].append(RelationGroup(group.batch, group.relation,
                group.sources[:max(degrees)], torch.tensor(nodes, dtype=torch.long),
                torch.tensor(degrees, dtype=torch.long)))
    windows = []
    history = torch.empty(0, dtype=torch.long)
    for (receivers, edges, present), local in zip(metadata, local_groups):
        plan = pack_attention_plan(timestamp, var_idx, node_type, valid,
            receiver_chunk=receiver_chunk, max_score_elements=max_score_elements,
            within_relations=within_relations, groups=local)
        layout, history = _compact_layout(plan, receivers, node_type, history)
        windows.append(WindowStep(plan, receivers, edges, present, layout))
    return PackedAttentionPlan([], tuple(within_relations), windows)


class WindowPatternMemory(nn.Module):
    """Read out known interactions, then query their ordered history for next state."""
    def __init__(self, dim, num_vars, heads):
        super().__init__()
        self.initial = nn.Parameter(torch.randn(dim) * .02)
        self.empty = nn.Parameter(torch.randn(dim) * .02)
        self.boundary = nn.Sequential(nn.Linear(3, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.state_norm = nn.LayerNorm(dim)
        self.state_ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.state_output_norm = nn.LayerNorm(dim)
        self.interactions = nn.Linear(3 * dim, dim)
        self.variable = nn.Embedding(num_vars, dim)
        self.role = nn.Embedding(4, dim)
        self.relative_time = nn.Sequential(nn.Linear(1, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.readout_norm = nn.LayerNorm(dim)
        self.readout_score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False))

    def _boundary(self, bounds):
        return self.boundary(torch.cat((bounds, bounds[:, 1:] - bounds[:, :1]), -1))

    def predict(self, tokens, token_valid, bounds):
        query = self.initial[None] + self._boundary(bounds)
        if tokens:
            history = torch.stack(tokens, 1)
            keep = torch.stack(token_valid, 1)
            # MHA cannot normalize a fully masked row. A zero dummy is ignored
            # afterwards, preserving empty/padded records without NaNs.
            any_history = keep.any(-1)
            safe = keep.clone()
            safe[:, 0] |= ~any_history
            history = torch.where(keep.unsqueeze(-1), history, 0)
            value, _ = self.attention(query[:, None], history, history,
                key_padding_mask=~safe, need_weights=False)
            query = query + torch.where(any_history[:, None], value[:, 0], 0)
        state = self.state_norm(query)
        return self.state_output_norm(state + self.state_ffn(state))

    def summarize(self, stages, timestamp, var_idx, node_type, known, bounds):
        # Unselected/query states are removed before learned arithmetic, including
        # zero-gradient edges: query tokens are never relation-memory sources.
        features = torch.cat([torch.where(known.unsqueeze(-1), h, 0) for h in stages], -1)
        relative = torch.where(known, timestamp - bounds[:, :1], 0)
        features = self.interactions(features) + self.variable(var_idx) + self.role(node_type)
        features = features + self.relative_time(relative.unsqueeze(-1))
        features = self.readout_norm(features)
        scores = self.readout_score(features).squeeze(-1)
        if known.shape[1] == 0:
            # Empty tensor operations preserve explicit zero gradients, matching
            # the reference even for optimizers with momentum/weight decay.
            zero = (features.sum() + scores.sum()) * 0
            return self.empty[None] + self._boundary(bounds) + zero
        weights = masked_softmax(scores, known, dim=-1)
        token = (features * weights.unsqueeze(-1)).sum(1)
        token = torch.where(known.any(-1, keepdim=True), token, self.empty[None])
        return token + self._boundary(bounds)
