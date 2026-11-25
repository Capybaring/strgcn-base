"""Voronoi density as an observed-X attention relation or node input.

The legacy irgraph implementation and checkpoints retain their original meaning.
Node variants use either the legacy concatenation/MLP or equal-width channels
concatenated directly into attention. Neither estimates density with a network
or adds a separate density attention bias.
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
        density_location="attention",
        node_encoding="mlp",
    ):
        super().__init__()
        if min(dim, heads, node_dim, layers, chunk_size) < 1 or dim % 2 or dim % heads:
            raise ValueError(
                "Graph Transformer requires even width divisible by heads and positive sizes"
            )
        self.use_density_encoding = use_density_encoding
        if density_location not in ("attention", "node"):
            raise ValueError("density_location must be attention or node")
        self.density_location = density_location
        if node_encoding not in ("mlp", "concat"):
            raise ValueError("node_encoding must be mlp or concat")
        if node_encoding == "concat" and density_location != "node":
            raise ValueError("concat node encoding requires density_location=node")
        self.node_encoding = node_encoding
        self.encoder = nn.Linear(1, dim)
        self.variable = nn.Embedding(num_vars, dim)
        self.time_embedding = TimeEmbedding(dim, "rbf")
        if density_location == "node" and node_encoding == "mlp":
            self.node_fusion = nn.Sequential(
                nn.Linear(3 * dim + 1, dim), nn.SiLU(), nn.Linear(dim, dim)
            )
        elif node_encoding == "concat":
            # Zero density must stay a zero vector, including regular grids.
            self.density_encoder = nn.Linear(1, dim, bias=False)
        self.layers = nn.ModuleList(
            [
                ExogenousGraphAttention(
                    dim,
                    heads,
                    num_vars,
                    node_dim,
                    chunk_size,
                    dropout,
                    density_relation=density_location == "attention",
                    density_projection=False,
                    input_dim=4 * dim if node_encoding == "concat" and i == 0 else dim,
                )
                for i in range(layers)
            ]
        )

    def forward(self, observations):
        n = observations
        density = voronoi_log_density(n.times, n.variables, n.valid)
        if not self.use_density_encoding:
            density = torch.zeros_like(density)
        p = time_features(self.time_embedding, n.times, n.valid)
        value, variable = self.encoder(n.values[..., None]), self.variable(n.variables)
        if self.node_encoding == "concat":
            h = torch.cat(
                [value, variable, p, self.density_encoder(density[..., None])], -1
            )
        elif self.density_location == "node":
            h = self.node_fusion(
                torch.cat([value, variable, p, density[..., None]], -1)
            )
        else:
            h = value + variable + p
        h = torch.where(n.valid[..., None], h, 0.0)
        for layer in self.layers:
            h = layer(h, p, n.variables, n.valid, density)
        return h
