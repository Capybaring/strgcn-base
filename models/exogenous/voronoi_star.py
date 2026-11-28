"""Receiver-centered Voronoi stars and density-conditioned vector attention.

Variable IDs are categorical. Geometry uses temporal distances only. The
reference is the arithmetic mean of rho over TRAINING receivers, conditional
on (receiver variable, source variable), and is frozen in checkpoint buffers.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.STRGCN.STRGCNDirect import time_features
from models.STRGCN.TimeEmbedding import TimeEmbedding


@dataclass
class VoronoiStar:
    log_rho: torch.Tensor  # [B,S,V], float64
    known: torch.Tensor  # [B,S,V], bool, a positive temporal cell exists
    neighbors: torch.Tensor  # [B,S,K], padded source token indices
    edge_valid: torch.Tensor  # [B,S,K]
    edge_type: torch.Tensor  # own/cross variable x left/right/synchronous


@torch.no_grad()
def voronoi_star(times, variables, valid, num_vars, build_edges=True):
    """Nearest strict left/right timestamp groups plus coincident observations.

    Both sides: L=(delta_left+delta_right)/2; one side: L=delta (mirror).
    Without a positive gap density is unknown. All duplicate values at a
    selected timestamp remain sources. Synchronous sources (including self)
    do not create zero-length cells. No dense S-by-S adjacency is allocated.
    """
    valid = valid.bool() & torch.isfinite(times)
    if ((variables[valid] < 0) | (variables[valid] >= num_vars)).any():
        raise ValueError("Observed variable ID outside num_vars")
    batch, size = times.shape
    log_rho = torch.zeros(
        (batch, size, num_vars), dtype=torch.float64, device=times.device
    )
    known = torch.zeros_like(log_rho, dtype=torch.bool)
    packed = []
    for b in range(batch):
        receivers = torch.where(valid[b])[0]
        query = times[b, receivers].double().contiguous()
        indices, masks, kinds = [], [], []
        for c in variables[b, receivers].unique(sorted=True):
            sources = torch.where(valid[b] & (variables[b] == c))[0]
            order = times[b, sources].argsort(stable=True)
            sources = sources[order]
            ticks, counts = (
                times[b, sources].double().unique_consecutive(return_counts=True)
            )
            lower = torch.searchsorted(ticks, query)
            upper = torch.searchsorted(ticks, query, right=True)
            left, right = lower - 1, upper
            has_left, has_right = left >= 0, right < len(ticks)
            dl = torch.where(
                has_left, query - ticks[left.clamp(0, len(ticks) - 1)], 0.0
            )
            dr = torch.where(
                has_right, ticks[right.clamp(0, len(ticks) - 1)] - query, 0.0
            )
            sides = has_left.long() + has_right.long()
            length = (dl + dr) / sides.clamp_min(1)
            supported = sides > 0
            log_rho[b, receivers, c] = -torch.where(supported, length, 1.0).log()
            known[b, receivers, c] = supported
            if not build_edges:
                continue
            starts = counts.cumsum(0) - counts
            offsets = torch.arange(int(counts.max()), device=times.device)
            coincident = (lower < len(ticks)) & (
                ticks[lower.clamp_max(len(ticks) - 1)] == query
            )
            for direction, group, exists in [
                (0, left, has_left),
                (1, right, has_right),
                (2, lower, coincident),
            ]:
                safe = group.clamp(0, len(ticks) - 1)
                pos = starts[safe, None] + offsets[None]
                edge_mask = exists[:, None] & (offsets[None] < counts[safe, None])
                source = sources[pos.clamp_max(len(sources) - 1)]
                idx = torch.zeros(
                    (size, len(offsets)), dtype=torch.long, device=times.device
                )
                mask = torch.zeros_like(idx, dtype=torch.bool)
                kind = torch.zeros_like(idx)
                idx[receivers], mask[receivers] = source, edge_mask
                kind[receivers] = (
                    (variables[b, receivers] != c).long() * 3 + direction
                )[:, None]
                indices.append(idx)
                masks.append(mask)
                kinds.append(kind)
        if indices:
            packed.append(tuple(torch.cat(x, -1) for x in (indices, masks, kinds)))
        else:
            idx = torch.zeros((size, 1), dtype=torch.long, device=times.device)
            packed.append((idx, torch.zeros_like(idx, dtype=torch.bool), idx.clone()))
    width = max(x[0].shape[-1] for x in packed)
    neighbor = torch.zeros((batch, size, width), dtype=torch.long, device=times.device)
    edge_valid = torch.zeros_like(neighbor, dtype=torch.bool)
    edge_type = torch.zeros_like(neighbor)
    for b, (idx, mask, kind) in enumerate(packed):
        neighbor[b, :, : idx.shape[-1]] = idx
        edge_valid[b, :, : idx.shape[-1]] = mask
        edge_type[b, :, : idx.shape[-1]] = kind
    return VoronoiStar(log_rho, known, neighbor, edge_valid, edge_type)


@torch.no_grad()
def relative_density_field(
    log_rho, known, receiver_ids, log_reference, reference_count
):
    """Return z=[s,a,S] and known mask; never impute unknown density as r=1."""
    known = known & (reference_count[receiver_ids] > 0)
    s = log_rho - log_reference[receiver_ids].double()
    s = torch.where(known, s, 0.0)
    count = known.sum(-1)
    log_count = count.clamp_min(1).double().log()
    log_sum = s.masked_fill(~known, -torch.inf).logsumexp(-1)
    log_inv_sum = (-s).masked_fill(~known, -torch.inf).logsumexp(-1)
    log_sum = torch.where(count > 0, log_sum, 0.0)
    global_level = torch.where(count > 0, log_count - log_inv_sum, 0.0)
    a = s + log_count[..., None] - log_sum[..., None]
    z = torch.stack([s, a, global_level[..., None].expand_as(s)], -1)
    return torch.where(known[..., None], z, 0.0), known


class VoronoiStarAttention(nn.Module):
    def __init__(self, dim, heads, node_dim, chunk_size, dropout):
        super().__init__()
        self.heads, self.width = heads, dim // heads
        self.node_dim, self.chunk_size = node_dim, chunk_size
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.time_scale = nn.Parameter(torch.ones(heads))
        self.var_scale = nn.Parameter(torch.full((heads,), 0.01))
        self.edge_kind = nn.Embedding(6, node_dim)
        self.edge_net = nn.Sequential(
            nn.Linear(2 * self.width + 3 + dim + node_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, self.width),
        )
        self.density_gate = nn.Parameter(torch.zeros(heads, self.width))
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h,
        p,
        relation_q,
        relation_k,
        ids,
        valid,
        star,
        z,
        known,
        context,
        use_density,
    ):
        batch, size, dim = h.shape
        split = lambda x: x.reshape(batch, size, self.heads, -1)
        n = self.norm1(h)
        q, k, v = (split(project(n)) for project in (self.q, self.k, self.v))
        p = split(p)
        rows = torch.arange(batch, device=h.device)[:, None, None]
        blocks = []
        for start in range(0, size, self.chunk_size):
            stop = min(start + self.chunk_size, size)

            # Bind offsets: recomputation must use this chunk during backward.
            def aggregate(q, k, v, p, rq, rk, z, context, start=start, stop=stop):
                ix = star.neighbors[:, start:stop]
                key, value = k[rows, ix], v[rows, ix]
                query = q[:, start:stop, None]
                base = (query * key).sum(-1) / math.sqrt(self.width)
                base = base + self.time_scale * (
                    p[:, start:stop, None] * p[rows, ix]
                ).sum(-1)
                base = base + self.var_scale * (
                    rq[:, start:stop, None] * rk[rows, ix]
                ).sum(-1) / math.sqrt(self.node_dim)
                logits = base[..., None].expand_as(value)
                if use_density:
                    source_ids = ids[rows, ix]
                    edge_z = z[:, start:stop].gather(
                        2, source_ids[..., None].expand(-1, -1, -1, 3)
                    )
                    supported = known[:, start:stop].gather(2, source_ids)
                    shape = (*ix.shape, self.heads, -1)
                    edge_input = torch.cat(
                        [
                            query.expand_as(key),
                            key,
                            edge_z[..., None, :].expand(*shape[:-1], 3),
                            context[:, start:stop, None, None, :].expand(
                                *shape[:-1], dim
                            ),
                            self.edge_kind(star.edge_type[:, start:stop])[
                                ..., None, :
                            ].expand(*shape[:-1], self.node_dim),
                        ],
                        -1,
                    )
                    bias = self.edge_net(edge_input) * supported[..., None, None]
                    logits = logits + self.density_gate * bias
                mask = star.edge_valid[:, start:stop, :, None, None]
                logits = logits.masked_fill(~mask, -torch.inf)
                logits = torch.where(mask.any(2, keepdim=True), logits, 0.0)
                weights = logits.softmax(2) * mask
                return (weights * value).sum(2).reshape(batch, stop - start, dim)

            args = (q, k, v, p, relation_q, relation_k, z, context)
            message = (
                checkpoint(aggregate, *args, use_reentrant=False)
                if self.training and torch.is_grad_enabled()
                else aggregate(*args)
            )
            blocks.append(message)
        h = h + self.dropout(self.out(torch.cat(blocks, 1)))
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return torch.where(valid[..., None], h, 0.0)


class VoronoiStarTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        heads=4,
        node_dim=16,
        layers=1,
        chunk_size=128,
        dropout=0.0,
        use_density_encoding=True,
    ):
        super().__init__()
        if (
            min(dim, num_vars, heads, node_dim, layers, chunk_size) < 1
            or dim % 2
            or dim % heads
        ):
            raise ValueError(
                "Voronoi star requires even width divisible by heads and positive sizes"
            )
        self.num_vars, self.dim, self.use_density_encoding = (
            num_vars,
            dim,
            use_density_encoding,
        )
        self.encoder = nn.Linear(1, dim)
        self.variable = nn.Embedding(num_vars, dim)
        self.time_embedding = TimeEmbedding(dim, "rbf")
        self.density_variable = nn.Embedding(num_vars, node_dim)
        self.context_net = nn.Sequential(
            nn.Linear(2 + node_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.relation_q, self.relation_k = (
            nn.Embedding(num_vars, heads * node_dim) for _ in range(2)
        )
        self.layers = nn.ModuleList(
            [
                VoronoiStarAttention(dim, heads, node_dim, chunk_size, dropout)
                for _ in range(layers)
            ]
        )
        self.register_buffer(
            "log_reference", torch.zeros(num_vars, num_vars, dtype=torch.float64)
        )
        self.register_buffer(
            "reference_count", torch.zeros(num_vars, num_vars, dtype=torch.long)
        )
        self.register_buffer("reference_ready", torch.tensor(False))

    @torch.no_grad()
    def fit_reference(self, observations):
        """Explicit calibration only; forward/eval never update the reference."""
        sums = torch.full(
            (self.num_vars, self.num_vars), -torch.inf, dtype=torch.float64
        )
        counts = torch.zeros_like(sums, dtype=torch.long)
        for obs in observations:
            times, ids, valid = (
                x.detach().cpu() for x in (obs.times, obs.variables, obs.valid)
            )
            star = voronoi_star(times, ids, valid, self.num_vars, build_edges=False)
            for a in ids[valid].unique():
                mask = star.known & (valid & (ids == a))[..., None]
                batch_sum = (
                    star.log_rho.masked_fill(~mask, -torch.inf)
                    .reshape(-1, self.num_vars)
                    .logsumexp(0)
                )
                sums[a] = torch.logaddexp(sums[a], batch_sum)
                counts[a] += mask.sum((0, 1))
        reference = torch.where(
            counts > 0, sums - counts.clamp_min(1).double().log(), 0.0
        )
        self.log_reference.copy_(reference)
        self.reference_count.copy_(counts)
        self.reference_ready.fill_(True)

    def forward(self, n):
        if self.use_density_encoding and not bool(self.reference_ready):
            raise RuntimeError(
                "Fit Voronoi density reference on training X before forward; it is saved in checkpoints"
            )
        valid = n.valid.bool() & torch.isfinite(n.times) & torch.isfinite(n.values)
        times = torch.where(valid, n.times, 0.0)
        ids = torch.where(valid, n.variables, 0).long()
        values = torch.where(valid, n.values, 0.0)
        star = voronoi_star(times, ids, valid, self.num_vars)
        p = time_features(self.time_embedding, times, valid)
        h = self.encoder(values[..., None]) + self.variable(ids) + p
        h = torch.where(valid[..., None], h, 0.0)
        context = h.new_zeros(h.shape)
        z = h.new_zeros((*star.known.shape, 3))
        known = torch.zeros_like(star.known)
        if self.use_density_encoding:
            z, known = relative_density_field(
                star.log_rho, star.known, ids, self.log_reference, self.reference_count
            )
            z = z.to(h.dtype)
            embeddings = self.density_variable.weight[None, None].expand(
                *z.shape[:-1], -1
            )
            eta = self.context_net(torch.cat([z[..., 1:], embeddings], -1))
            context = (eta * known[..., None]).sum(2) / known.sum(
                2, keepdim=True
            ).clamp_min(1)
        relation_q = self.relation_q(ids).reshape(*ids.shape, self.layers[0].heads, -1)
        relation_k = self.relation_k(ids).reshape_as(relation_q)
        for layer in self.layers:
            h = layer(
                h,
                p,
                relation_q,
                relation_k,
                ids,
                valid,
                star,
                z,
                known,
                context,
                self.use_density_encoding,
            )
        return h


def calibrate_voronoi_references(model, data_loader):
    """Separate deterministic training-only pass; no model forward or Y reads."""
    modules = [
        m
        for m in model.modules()
        if isinstance(m, VoronoiStarTransformer) and not bool(m.reference_ready)
    ]
    if not modules:
        return
    from torch.utils.data import DataLoader

    from .adapter import PatchQueryFusion

    loader = DataLoader(
        data_loader.dataset,
        batch_size=data_loader.batch_size or 1,
        shuffle=False,
        collate_fn=data_loader.collate_fn,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )

    def observations():
        for b in loader:
            source = b["batch_pad_mask"].bool() & (
                (b["batch_node_type"] == 0) | (b["batch_node_type"] == 2)
            )
            source = source & ~b["batch_pred_mask"].bool()
            source = (
                source
                & torch.isfinite(b["batch_timestamp"])
                & torch.isfinite(b["batch_value"])
            )
            yield PatchQueryFusion._observations(b, source)

    for module in modules:
        module.fit_reference(observations())
