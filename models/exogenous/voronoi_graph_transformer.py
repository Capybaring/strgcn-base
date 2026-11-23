"""Voronoi density as an independent observed-X graph attention relation.

The legacy irgraph implementation and checkpoints retain their original meaning.
No node MLP, density Q/K projection or X-to-Y density injection is used here.
"""

import torch
from torch import nn

from models.STRGCN.STRGCNDirect import time_features
from models.STRGCN.TimeEmbedding import TimeEmbedding
from .irregular_graph_transformer import ExogenousGraphAttention


@torch.no_grad()
def voronoi_log_density(times, variables, valid):
    """Return s=log(mu*rho), shape [B,S], independently per sample/variable.

    Interior rho=2/(delta_left+delta_right). Endpoints use rho=1/delta
    (symmetric continuation). A singleton has unknown density, encoded as zero.
    Geometry uses unique times, retaining every duplicate observation token.
    Float64 arithmetic and the legacy input-resolution tolerance ensure exact
    zero on representably uniform grids. No clipping or learned density map.
    """
    density = torch.zeros_like(times)
    eps = torch.finfo(times.dtype).eps
    for b in range(times.shape[0]):
        for variable in variables[b, valid[b]].unique():
            ix = torch.where(valid[b] & (variables[b] == variable))[0]
            t, inverse = times[b, ix].double().unique(sorted=True, return_inverse=True)
            if len(t) < 2:
                continue
            gaps = t.diff()
            mu = gaps.mean()
            left = torch.cat([gaps.new_zeros(1), gaps])
            right = torch.cat([gaps, gaps.new_zeros(1)])
            count = torch.full_like(t, 2)
            count[0] = count[-1] = 1
            # Logs avoid overflowing rho=1/length for near-coincident times.
            s = mu.log() + count.log() - (left + right).log()
            tolerance = 2 * eps * torch.maximum(t.abs().max(), mu)
            uniform = ((gaps - mu).abs() <= tolerance).all()
            s = torch.where(uniform, 0.0, s)
            density[b, ix] = s[inverse].to(times.dtype)
    return density


class VoronoiGraphTransformer(nn.Module):
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
        if min(dim, heads, node_dim, layers, chunk_size) < 1 or dim % 2 or dim % heads:
            raise ValueError(
                "Graph Transformer requires even width divisible by heads and positive sizes"
            )
        self.use_density_encoding = use_density_encoding
        self.encoder = nn.Linear(1, dim)
        self.variable = nn.Embedding(num_vars, dim)
        self.time_embedding = TimeEmbedding(dim, "rbf")
        self.layers = nn.ModuleList(
            [
                ExogenousGraphAttention(
                    dim,
                    heads,
                    num_vars,
                    node_dim,
                    chunk_size,
                    dropout,
                    density_relation=True,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, observations):
        n = observations
        density = voronoi_log_density(n.times, n.variables, n.valid)
        if not self.use_density_encoding:
            density = torch.zeros_like(density)
        p = time_features(self.time_embedding, n.times, n.valid)
        h = self.encoder(n.values[..., None]) + self.variable(n.variables) + p
        h = torch.where(n.valid[..., None], h, 0.0)
        for layer in self.layers:
            h = layer(h, p, n.variables, n.valid, density)
        return h
