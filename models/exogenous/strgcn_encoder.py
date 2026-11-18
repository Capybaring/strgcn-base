"""X-only STRGCN middle-stage operator, without Sandwich or time direction masks.

Reuse the original joint relation logits, Chebyshev recurrence, time/variable
weight factorization and residual update through DirectSTRGCNLayer. Only the
support's visibility differs: every valid supplied X can send to every X.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.STRGCN.STRGCNDirect import DirectSTRGCNLayer, time_features
from models.STRGCN.STRGCN_Layer import masked_softmax
from models.STRGCN.TimeEmbedding import TimeEmbedding


class UnrestrictedSTRGCNLayer(DirectSTRGCNLayer):
    def support_apply(self, x, p, vars_idx, tt, valid, query, plan):
        u = self.learnable_adj_weight_U(vars_idx)
        v = self.learnable_adj_weight_V(vars_idx)
        time_scale = self.middle_time_temp * self.middle_time_scale
        var_scale = self.middle_val_temp * self.middle_val_scale
        blocks = []
        for start in range(0, x.shape[1], self.chunk_size):
            end = min(start + self.chunk_size, x.shape[1])

            def aggregate(p, x, u, v, ts, vs, start=start, end=end):
                logits = ts * (p[:, start:end] @ p.transpose(-1, -2))
                logits = logits + vs * (u[:, start:end] @ v.transpose(-1, -2))
                mask = valid[:, start:end, None] & valid[:, None, :]
                return masked_softmax(logits, mask) @ x

            args = (p, x, u, v, time_scale, var_scale)
            if self.training and torch.is_grad_enabled():
                message = checkpoint(aggregate, *args, use_reentrant=False)
            else:
                message = aggregate(*args)
            blocks.append(message)
        return torch.cat(blocks, 1)


class STRGCNExogenousEncoder(nn.Module):
    def __init__(
        self, dim, num_vars, node_dim=16, depth=2, layers=1, chunk_size=128, dropout=0.0
    ):
        super().__init__()
        if dim % 2 or min(node_dim, depth, layers, chunk_size) < 1:
            raise ValueError("STRGCN requires even hidden width and positive sizes")
        self.encoder = nn.Linear(1, dim)
        self.time_embedding = TimeEmbedding(dim, "rbf")
        self.layers = nn.ModuleList(
            [
                UnrestrictedSTRGCNLayer(
                    dim,
                    node_dim,
                    num_vars,
                    dropout,
                    True,
                    True,
                    "rbf",
                    "cheb",
                    depth,
                    "dense",
                    chunk_size,
                )
                for _ in range(layers)
            ]
        )
        self.lambda_ = nn.Parameter(torch.tensor(0.5))

    def forward(self, observations):
        n = observations  # callers sanitize invalid values/times/IDs first
        valid = n.valid.bool()
        h0 = self.encoder(n.values[..., None]) + time_features(
            self.time_embedding, n.times, valid
        )
        h0 = torch.where(valid[..., None], h0, 0.0)
        h = h0
        query = torch.zeros_like(valid)
        for layer in self.layers:
            h = layer(h, n.variables, n.times, valid, query, None)
        return torch.where(
            valid[..., None], self.lambda_ * h + (1 - self.lambda_) * h0, 0.0
        )
