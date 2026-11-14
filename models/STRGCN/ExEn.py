"""Observation-level Ex-En baseline, without Sandwich or hypernodes.

All adjacency tensors use [batch, receiver, sender] ordering.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .SparseRelations import build_sparse_relations, sparse_messages
from .TimeEmbedding import TimeEmbedding
from .STRGCN_Layer import low_rank_graph_conv, masked_softmax

HIST_EXO, HIST_ENDO, FUTURE_EXO, QUERY_ENDO = range(4)


def relation_masks(timestamp, var_idx, node_type, valid, receiver_slice=slice(None)):
    """Return X-time, Y-time, X->Y masks. Queries NEVER send messages."""
    valid = valid.bool()
    exo = (node_type == HIST_EXO) | (node_type == FUTURE_EXO)
    endo = (node_type == HIST_ENDO) | (node_type == QUERY_ENDO)
    # j <= i; equal timestamps are allowed, as in the method definition.
    causal = timestamp[:, None, :] <= timestamp[:, receiver_slice, None]
    source = valid & (node_type != QUERY_ENDO)
    base = causal & valid[:, receiver_slice, None] & source[:, None, :]
    same = var_idx[:, receiver_slice, None] == var_idx[:, None, :]
    return (
        base & same & exo[:, receiver_slice, None] & exo[:, None, :],
        base & same & endo[:, receiver_slice, None] & endo[:, None, :],
        base & endo[:, receiver_slice, None] & exo[:, None, :],
    )


class DirectedRelationLayer(nn.Module):
    """Reuse the efficient STRGCN relation logits and low-rank convolution.

    Temporal edges use time logits only; X->Y adds U(receiver).V(sender).
    Masking is BEFORE normalization, including empty neighborhoods.
    """
    def __init__(self, dim, node_dim, num_vars, dropout, use_bias, use_norm,
                 cross=False):
        super().__init__()
        self.cross = cross
        count = 1 if cross else 2
        self.weight_T = nn.ModuleList([nn.Linear(dim, dim) for _ in range(count)])
        self.weight_S = nn.ModuleList([nn.Embedding(num_vars, dim) for _ in range(count)])
        self.time_scale = nn.Parameter(torch.ones(count))
        if cross:
            self.adj_U = nn.Embedding(num_vars, node_dim)
            self.adj_V = nn.Embedding(num_vars, node_dim)
            self.var_scale = nn.Parameter(torch.tensor(1.0))
        self.self_linear = nn.Linear(dim, dim, bias=use_bias)
        self.lambda_ = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()

    def _aggregate(self, h, p, var_idx, masks, receiver_slice=slice(None)):
        time_logits = torch.einsum('bid,bjd->bij', p[:, receiver_slice], p)
        message = torch.zeros_like(h[:, receiver_slice])
        for k, mask in enumerate(masks):
            logits = self.time_scale[k] * time_logits
            if self.cross:
                support = torch.einsum('bid,bjd->bij', self.adj_U(var_idx[:, receiver_slice]),
                                       self.adj_V(var_idx))
                logits = logits + self.var_scale * support / math.sqrt(self.adj_U.embedding_dim)
            adj = masked_softmax(logits, mask)
            message = message + low_rank_graph_conv(
                h, adj, self.weight_T[k](p), self.weight_S[k](var_idx))
        return message, masks[0].any(-1)

    def forward(self, h, p, var_idx, masks, valid, timestamp=None, node_type=None,
                receiver_chunk_size=None, sparse_plan=None):
        if sparse_plan is not None:
            message, receiver = sparse_messages(self, h, p, var_idx, sparse_plan, receiver_chunk_size or 128)
        elif receiver_chunk_size is None:
            message, receiver = self._aggregate(h, p, var_idx, masks)
        else:
            messages, receivers = [], []
            for start in range(0, h.shape[1], receiver_chunk_size):
                end = min(start + receiver_chunk_size, h.shape[1])
                def aggregate(h, p, start=start, end=end):
                    block = slice(start, end)
                    block_masks = relation_masks(timestamp, var_idx, node_type, valid, block)
                    selected = block_masks[2:] if self.cross else block_masks[:2]
                    return self._aggregate(h, p, var_idx, selected, block)
                # Recompute block intermediates in backward; do not retain L x L activations.
                if self.training and torch.is_grad_enabled():
                    msg, recv = checkpoint(aggregate, h, p, use_reentrant=False)
                else:
                    msg, recv = aggregate(h, p)
                messages.append(msg)
                receivers.append(recv)
            message, receiver = torch.cat(messages, 1), torch.cat(receivers, 1)
        updated = F.relu(message + self.self_linear(h))
        out = self.norm(self.lambda_ * self.dropout(updated) + (1 - self.lambda_) * h)
        # Cross stage updates only receivers with an endogenous role. X is fixed.
        if self.cross:
            # Empty-neighborhood receivers still retain their prior representation.
            out = torch.where(receiver.unsqueeze(-1), out, h)
        return torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))


class ExEn(nn.Module):
    """Four explicit node types; same relation parameters for history and future.

    Forward returns [B,L,1], compatible with IrrFCGraphForecastTrainer.
    Query values may contain labels/NaN: they are removed BEFORE any arithmetic.
    """
    def __init__(self, c_in=1, c_out=1, hid_dim=16, n_layer=2, node_dim=16,
                 dropout=0.0, use_bias=True, use_norm=True,
                 time_embedding_type='rbf', global_args=None, num_vars=None,
                 n_cross_layer=1, exogenous_var_indices=None,
                 gcn_type='gcn', gcn_depth=1, use_efficient_gcn=False,
                 use_info_agg=False, use_revin=False, receiver_chunk_size=None, relation_backend="dense", **kwargs):
        super().__init__()
        if c_in != 1 or c_out != 1:
            raise ValueError('ExEn expects scalar observation nodes (c_in=c_out=1)')
        if use_efficient_gcn or use_info_agg or use_revin:
            raise ValueError('ExEn does not use Sandwich, aggregation or RevIN')
        self._validate_operator(gcn_type, gcn_depth)
        if time_embedding_type not in ('rbf', 'rope', 'transformer') or hid_dim % 2:
            raise ValueError('Use an even hid_dim and rbf/rope/transformer time embedding')
        self._validate_depths(n_layer, n_cross_layer)
        global_args = global_args or {}
        if receiver_chunk_size is not None and (not isinstance(receiver_chunk_size, int) or receiver_chunk_size < 1):
            raise ValueError('receiver_chunk_size must be a positive integer or null')
        if relation_backend not in ("dense", "sparse_csr"):
            raise ValueError("relation_backend must be dense or sparse_csr")
        self.relation_backend = relation_backend
        self.receiver_chunk_size = receiver_chunk_size
        self.num_vars = num_vars or global_args.get('num_of_vals')
        if not self.num_vars:
            raise ValueError('Supply num_vars or global_args[num_of_vals]')
        indices = exogenous_var_indices
        if indices is None:
            indices = global_args.get('exogenous_var_indices')
        if indices is None or not indices or len(set(indices)) != len(indices):
            raise ValueError('Supply unique, nonempty exogenous_var_indices')
        if any(i < 0 or i >= self.num_vars for i in indices) or len(indices) >= self.num_vars:
            raise ValueError('Need valid exogenous indices and at least one endogenous variable')
        is_exo = torch.zeros(self.num_vars, dtype=torch.bool)
        is_exo[indices] = True
        self.register_buffer('is_exo', is_exo)
        self.encoder = nn.Linear(c_in, hid_dim)
        self.pred_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.time_embedding = TimeEmbedding(hid_dim, time_embedding_type)
        self.role_embedding = nn.Embedding(2, hid_dim)
        self.status_embedding = nn.Embedding(3, hid_dim)
        args = (hid_dim, node_dim, self.num_vars, dropout, use_bias, use_norm)
        self._relation_layer_args = args
        self.temporal_layers = nn.ModuleList([self._make_relation_layer(*args) for _ in range(n_layer)])
        self.cross_layers = nn.ModuleList([self._make_relation_layer(*args, cross=True) for _ in range(n_cross_layer)])
        self.lambda_ = nn.Parameter(torch.tensor(0.5))
        # Original STRGCN decoder is a one-layer MLP (nn.Linear), preserved here.
        self.projection = nn.Linear(hid_dim, c_out)

    def _validate_operator(self, gcn_type, gcn_depth):
        if gcn_type != 'gcn' or gcn_depth != 1:
            raise ValueError('ExEn uses direct masked gcn (gcn_depth=1); stack n_layer instead')

    def _make_relation_layer(self, *args, cross=False):
        return DirectedRelationLayer(*args, cross=cross)

    def _validate_depths(self, n_layer, n_cross_layer):
        if n_layer < 1 or n_cross_layer < 1:
            raise ValueError('Both stage depths must be positive')

    def _apply_variable_layers(self, h, p, var_idx, valid, timestamp, node_type, plan):
        return h

    def forward(self, batch_value, batch_timestamp, batch_var_idx,
                batch_pred_mask=None, batch_pad_mask=None, batch_time_id=None,
                batch_agg_mask=None, batch_node_type=None, batch_attention_plan=None):
        if batch_node_type is None:
            raise ValueError('batch_node_type is required; never infer roles from target values')
        if batch_value.ndim != 2 or batch_value.shape[1] == 0:
            raise ValueError('Node tensors must have nonempty shape [B,L]')
        valid = torch.ones_like(batch_value, dtype=torch.bool) if batch_pad_mask is None else batch_pad_mask.bool()
        if any(t.shape != batch_value.shape for t in (batch_timestamp, batch_var_idx, batch_node_type, valid)):
            raise ValueError('All node tensors must have shape [B,L]')
        for metadata in (batch_var_idx, batch_node_type):
            selected = metadata[valid]
            if not torch.isfinite(selected).all() or (selected != selected.long()).any():
                raise ValueError('Variable indices and node types must be finite integers')
        if ((batch_var_idx[valid] < 0) | (batch_var_idx[valid] >= self.num_vars)).any():
            raise ValueError('Variable index out of range')
        if ((batch_node_type[valid] < 0) | (batch_node_type[valid] > 3)).any():
            raise ValueError('Node types must be in 0..3')
        var_idx = torch.where(valid, batch_var_idx, 0).long()
        types = torch.where(valid, batch_node_type, 0).long()
        exo = (types == HIST_EXO) | (types == FUTURE_EXO)
        if (self.is_exo[var_idx][valid] != exo[valid]).any():
            raise ValueError('Node types disagree with configured variable roles')
        query = (types == QUERY_ENDO) & valid
        if batch_pred_mask is not None and not torch.equal(batch_pred_mask.bool() & valid, query):
            raise ValueError('Prediction mask must select exactly the valid endogenous queries')
        tt = torch.where(valid, batch_timestamp, 0)
        values = torch.where(valid & ~query, batch_value, 0)
        if not torch.isfinite(values).all() or not torch.isfinite(tt).all():
            raise ValueError('Known observations and valid timestamps must be finite')
        # Validate a common forecast origin: no future node may precede history.
        history = valid & ((types == HIST_EXO) | (types == HIST_ENDO))
        future = valid & ~history
        latest_history = tt.masked_fill(~history, -torch.inf).amax(-1)
        first_future = tt.masked_fill(~future, torch.inf).amin(-1)
        if (latest_history >= first_future).any():
            raise ValueError('History timestamps must be strictly before future timestamps')
        p = self.time_embedding(tt, valid)
        if p.ndim == 2:  # upstream TimeEmbedding squeezes singleton batches
            p = p.unsqueeze(0)
        encoded = self.encoder(values.unsqueeze(-1)) + p
        h0 = torch.where(query.unsqueeze(-1), self.pred_token, encoded)
        status = torch.where(types == FUTURE_EXO, 1, torch.where(query, 2, 0))
        h0 = h0 + self.role_embedding((~exo).long()) + self.status_embedding(status)
        h0 = torch.where(valid.unsqueeze(-1), h0, 0)
        sparse_plan = batch_attention_plan if batch_attention_plan is not None else (
            build_sparse_relations(tt, var_idx, types, valid) if self.relation_backend == "sparse_csr" else None)
        masks = relation_masks(tt, var_idx, types, valid) if sparse_plan is None and self.receiver_chunk_size is None else None
        h = self._run_graph_layers(h0, p, var_idx, valid, tt, types, sparse_plan, masks)
        result = self.projection(self.lambda_ * h + (1 - self.lambda_) * h0)
        return torch.where(valid.unsqueeze(-1), result, 0)

    def _run_graph_layers(self, h, p, var_idx, valid, tt, types, sparse_plan, masks):
        for layer in self.temporal_layers:
            h = layer(h, p, var_idx, masks[:2] if masks is not None else None, valid,
                      tt, types, self.receiver_chunk_size, sparse_plan=sparse_plan)
        h = self._apply_variable_layers(h, p, var_idx, valid, tt, types, sparse_plan)
        for layer in self.cross_layers:
            h = layer(h, p, var_idx, masks[2:] if masks is not None else None, valid,
                      tt, types, self.receiver_chunk_size, sparse_plan=sparse_plan)
        return h
