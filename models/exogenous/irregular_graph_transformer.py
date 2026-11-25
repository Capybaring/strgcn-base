"""Observed-X graph attention with regular-grid-zero sampling geometry.

Geometry uses only within-variable timestamps, independently per sample. It
never creates interpolated observations or uses Y labels. All supplied X are
visible, including sources later than a receiving X or Y query.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.STRGCN.STRGCNDirect import time_features
from models.STRGCN.TimeEmbedding import TimeEmbedding


def voronoi_density_bias(receivers, sources, log_scale):
    """Independent [B,H,Q,S] log-Gaussian bias; regular/regular is exact zero."""
    difference = receivers[:, None, :, None] - sources[:, None, None, :]
    return (
        -torch.nn.functional.softplus(log_scale)[None, :, None, None]
        * difference.square()
    )


@torch.no_grad()
def sampling_geometry(times, variables, valid):
    """Return [B,S,2] relative gaps, edge-existence flags and log densities.

    z=(delta-m*mu)/mu; density=m*log(mu/delta). A missing edge has m=0,
    delta=0, z=0 and density=0. Duplicate timestamps share geometry, but no
    observation/value is removed. Single unique times have no geometry.
    Float64 gap arithmetic and an input-precision rounding tolerance snap
    numerically uniform sequences to exact zero (including their boundaries).
    """
    z = times.new_zeros((*times.shape, 2))
    edges = torch.zeros_like(z)
    density = torch.zeros_like(z)
    eps = torch.finfo(times.dtype).eps
    for b in range(times.shape[0]):
        for variable in variables[b, valid[b]].unique():
            ix = torch.where(valid[b] & (variables[b] == variable))[0]
            t, inverse = times[b, ix].double().unique(sorted=True, return_inverse=True)
            if len(t) < 2:
                continue
            gaps = t.diff()
            mu = gaps.mean()
            m = torch.ones((len(t), 2), dtype=torch.float64, device=times.device)
            m[0, 0], m[-1, 1] = 0, 0
            delta = torch.stack(
                [
                    torch.cat([gaps.new_zeros(1), gaps]),
                    torch.cat([gaps, gaps.new_zeros(1)]),
                ],
                -1,
            )
            relative = (delta - m * mu) / mu
            tolerance = 2 * eps * torch.maximum(t.abs().max(), mu)
            uniform = ((gaps - mu).abs() <= tolerance).all()
            relative = torch.where(uniform, 0.0, relative)
            # Compute the log on positive gaps, before any downcast to float32.
            safe_delta = torch.where(m.bool(), delta, mu)
            log_density = (mu / safe_delta).log()
            log_density = torch.where(uniform, 0.0, log_density).clamp(-8, 8)
            z[b, ix] = relative[inverse].to(times.dtype)
            edges[b, ix] = m[inverse].to(times.dtype)
            density[b, ix] = log_density[inverse].to(times.dtype)
    return z, edges, density


class CenteredGapEncoding(nn.Module):
    """Deterministic F(z,m)-F(0,m), including cancellation of learned biases."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, z, edges):
        return self.net(torch.cat([z, edges], -1)) - self.net(
            torch.cat([torch.zeros_like(z), edges], -1)
        )


class ExogenousGraphAttention(nn.Module):
    """Multihead content + continuous-time + variable-relation attention."""

    def __init__(
        self,
        dim,
        heads,
        num_vars,
        node_dim,
        chunk_size,
        dropout,
        density_relation=False,
        density_projection=True,
        input_dim=None,
    ):
        super().__init__()
        self.heads, self.width = heads, dim // heads
        self.chunk_size = chunk_size
        input_dim = dim if input_dim is None else input_dim
        self.norm1, self.norm2 = nn.LayerNorm(input_dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(input_dim, dim) for _ in range(3))
        # A concatenated first-layer input can go straight into Q/K/V.
        # Match only the parallel residual path to the attention output width.
        self.residual = (
            nn.Linear(input_dim, dim, bias=False) if input_dim != dim else nn.Identity()
        )
        self.density_relation = density_relation
        self.density_projection = density_projection and not density_relation
        if density_relation:
            # softplus(log(expm1(1))) = 1, independently for each head.
            self.density_log_scale = nn.Parameter(
                torch.full((heads,), math.log(math.expm1(1.0)))
            )
        elif self.density_projection:
            self.density_q, self.density_k = (
                nn.Linear(2, dim, bias=False) for _ in range(2)
            )
        self.u, self.w = (nn.Embedding(num_vars, heads * node_dim) for _ in range(2))
        self.node_dim = node_dim
        self.time_scale = nn.Parameter(torch.ones(heads))
        self.var_scale = nn.Parameter(torch.full((heads,), 0.01))
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def split(self, x):
        return x.reshape(*x.shape[:2], self.heads, -1).transpose(1, 2)

    def forward(self, h, time_embedding, ids, valid, density):
        n = self.norm1(h)
        q = self.split(self.q(n))
        k = self.split(self.k(n))
        if self.density_projection:
            q = q + self.split(self.density_q(density))
            k = k + self.split(self.density_k(density))
        v, p = self.split(self.v(n)), self.split(time_embedding)
        u, w = self.split(self.u(ids)), self.split(self.w(ids))
        blocks = []
        for start in range(0, h.shape[1], self.chunk_size):
            end = min(start + self.chunk_size, h.shape[1])

            def aggregate(q, k, v, p, u, w, ts, vs, density, ds, start=start, end=end):
                score = q[:, :, start:end] @ k.transpose(-1, -2) / math.sqrt(self.width)
                score = score + ts[None, :, None, None] * (
                    p[:, :, start:end] @ p.transpose(-1, -2)
                )
                score = score + vs[None, :, None, None] * (
                    u[:, :, start:end] @ w.transpose(-1, -2)
                ) / math.sqrt(self.node_dim)
                if self.density_relation:
                    score = score + voronoi_density_bias(
                        density[:, start:end], density, ds
                    )
                mask = valid[:, None, start:end, None] & valid[:, None, None, :]
                score = score.masked_fill(~mask, -torch.inf)
                score = torch.where(mask.any(-1, keepdim=True), score, 0.0)
                return (score.softmax(-1) * mask) @ v

            ds = self.density_log_scale if self.density_relation else self.time_scale
            args = (q, k, v, p, u, w, self.time_scale, self.var_scale, density, ds)
            if self.training and torch.is_grad_enabled():
                message = checkpoint(aggregate, *args, use_reentrant=False)
            else:
                message = aggregate(*args)
            blocks.append(message)
        message = (
            torch.cat(blocks, 2)
            .transpose(1, 2)
            .reshape(*h.shape[:2], self.heads * self.width)
        )
        h = self.residual(h) + self.dropout(self.out(message))
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return torch.where(valid[..., None], h, 0.0)


class IrregularGraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        heads=4,
        node_dim=16,
        layers=1,
        chunk_size=128,
        dropout=0.0,
        use_gap_encoding=True,
        use_density_encoding=True,
    ):
        super().__init__()
        if min(dim, heads, node_dim, layers, chunk_size) < 1 or dim % 2 or dim % heads:
            raise ValueError(
                "Graph Transformer requires even width divisible by heads and positive sizes"
            )
        self.use_gap_encoding = use_gap_encoding
        self.use_density_encoding = use_density_encoding
        self.encoder = nn.Linear(1, dim)
        self.variable = nn.Embedding(num_vars, dim)
        self.time_embedding = TimeEmbedding(dim, "rbf")
        self.gap_encoding = CenteredGapEncoding(dim)
        self.cross_density_key = nn.Linear(2, dim, bias=False)
        self.layers = nn.ModuleList(
            [
                ExogenousGraphAttention(
                    dim, heads, num_vars, node_dim, chunk_size, dropout
                )
                for _ in range(layers)
            ]
        )

    def encode_with_geometry(self, n):
        # Fusion sanitizes invalid values/times/IDs before calling this method.
        z, edges, density = sampling_geometry(n.times, n.variables, n.valid)
        p = time_features(self.time_embedding, n.times, n.valid)
        h = self.encoder(n.values[..., None]) + self.variable(n.variables) + p
        if self.use_gap_encoding:
            h = h + self.gap_encoding(z, edges)
        if not self.use_density_encoding:
            density = torch.zeros_like(density)
        h = torch.where(n.valid[..., None], h, 0.0)
        for layer in self.layers:
            h = layer(h, p, n.variables, n.valid, density)
        return h, density

    def forward(self, n):
        return self.encode_with_geometry(n)[0]
