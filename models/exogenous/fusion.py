"""Query-level ExEn XY attention and historical dynamic lag transfer."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .irregular_graph_transformer import IrregularGraphTransformer
from .strgcn_encoder import STRGCNExogenousEncoder
from .voronoi_graph_transformer import VoronoiGraphTransformer


@dataclass
class Observations:
    values: torch.Tensor  # [B,S], scaled observations, never query labels
    times: torch.Tensor  # [B,S], common normalized physical time
    variables: torch.Tensor  # [B,S], global IDs
    valid: torch.Tensor  # [B,S], observed zero is valid


class DynamicLagFusion(nn.Module):
    def __init__(
        self,
        query_dim,
        num_variables,
        hidden=32,
        heads=4,
        mode="v1",
        lag_bins=8,
        lag_max=1.0,
        chunk_size=128,
        dropout=0.0,
        use_time_bias=True,
        use_variable_bias=True,
        transfer="dynamic",
        x_encoder="pointwise",
        allow_future_x=False,
        x_layers=1,
        x_depth=2,
        x_node_dim=16,
        use_gap_encoding=True,
        use_density_encoding=True,
        density_location="attention",
        node_encoding="mlp",
    ):
        super().__init__()
        if mode not in ("cross", "v1", "v2") or transfer not in (
            "dynamic",
            "static",
            "none",
        ):
            raise ValueError("Invalid fusion mode/transfer")
        if (
            hidden < 1
            or heads < 1
            or hidden % heads
            or lag_bins < 2
            or lag_max <= 0
            or chunk_size < 1
        ):
            raise ValueError("Invalid attention/lag dimensions")
        self.hidden, self.heads, self.dim = hidden, heads, hidden // heads
        self.mode, self.transfer, self.chunk_size = mode, transfer, chunk_size
        self.use_time_bias, self.use_variable_bias = use_time_bias, use_variable_bias
        if x_encoder not in ("pointwise", "strgcn", "irgraph", "voronoi"):
            raise ValueError("x_encoder must be pointwise, strgcn, irgraph or voronoi")
        if x_encoder != "pointwise" and not allow_future_x:
            raise ValueError("Unrestricted X encoding requires allow_future_x=True")
        if x_encoder != "pointwise" and mode == "v2":
            raise ValueError(
                "Contextual X encoding currently supports cross/v1, not V2 transfer"
            )
        self.allow_future_x, self.x_encoder_type = allow_future_x, x_encoder
        self.query = nn.Linear(query_dim, hidden)
        self.value_enc = nn.Linear(1, hidden)
        self.variable = nn.Embedding(num_variables, hidden)
        self.time = nn.Linear(1, hidden)
        self.q, self.k, self.v = (nn.Linear(hidden, hidden) for _ in range(3))
        if mode != "cross" and use_variable_bias:
            self.u, self.w = (nn.Embedding(num_variables, hidden) for _ in range(2))
            self.var_scale = nn.Parameter(torch.full((heads,), 0.01))
        if mode != "cross" and use_time_bias:
            self.time_scale = nn.Parameter(torch.ones(heads))
        if mode == "v2" and transfer != "none":
            self.transfer_scale = nn.Parameter(torch.ones(heads))
        if mode == "v2" and transfer == "static":
            self.static_lag = nn.Parameter(torch.zeros(num_variables, heads, lag_bins))
        self.out = nn.Linear(hidden, query_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("centers", torch.linspace(0, lag_max, lag_bins))
        self.lag_width = lag_max / (lag_bins - 1)
        self.num_variables = num_variables
        self.x_context = (
            STRGCNExogenousEncoder(
                hidden,
                num_variables,
                x_node_dim,
                x_depth,
                x_layers,
                chunk_size,
                dropout,
            )
            if x_encoder == "strgcn"
            else None
        )
        if x_encoder == "irgraph":
            self.x_context = IrregularGraphTransformer(
                hidden,
                num_variables,
                heads,
                x_node_dim,
                x_layers,
                chunk_size,
                dropout,
                use_gap_encoding,
                use_density_encoding,
            )
        elif x_encoder == "voronoi":
            self.x_context = VoronoiGraphTransformer(
                hidden,
                num_variables,
                heads,
                x_node_dim,
                x_layers,
                chunk_size,
                dropout,
                use_density_encoding,
                density_location,
                node_encoding,
            )

    def protocol(self):
        is_gcn = self.x_encoder_type == "strgcn"
        is_graph = self.x_encoder_type == "irgraph"
        is_voronoi = self.x_encoder_type == "voronoi"
        is_density_node = is_voronoi and self.x_context.density_location == "node"
        is_concat = is_density_node and self.x_context.node_encoding == "concat"
        return dict(
            x_encoder=self.x_encoder_type,
            x_visibility="all supplied observed X"
            if self.allow_future_x
            else "source_time <= query_time",
            x_internal_visibility="all supplied observed X"
            if self.x_context is not None
            else "pointwise",
            mask="valid observations only; queries never sources"
            if self.allow_future_x
            else "physical source_time <= query_time; queries never sources",
            x_operator="STRGCN middle-stage joint relations and low-rank Chebyshev; no Sandwich"
            if is_gcn
            else (
                "multihead content/time/variable/density Graph Transformer"
                if is_voronoi and not is_density_node
                else "multihead content/time/variable Graph Transformer"
                if is_graph or is_density_node
                else "value + variable + sine-time embedding"
            ),
            x_layers=len(self.x_context.layers) if self.x_context is not None else 0,
            x_chebyshev_terms=self.x_context.layers[0].gcn_depth if is_gcn else 0,
            x_relation_rank=self.x_context.layers[
                0
            ].learnable_adj_weight_U.embedding_dim
            if is_gcn
            else 0,
            **(
                dict(
                    gap_node_encoding=self.x_context.use_gap_encoding,
                    density_attention_encoding=self.x_context.use_density_encoding,
                    sampling_geometry="per-sample per-X-variable unique observed times; left/right relative gaps",
                    regular_zero="z=(delta-m*mean_gap)/mean_gap; F(z,m)-F(0,m)",
                    density="left/right log(mean_gap/delta), clipped [-8,8]; missing edges zero",
                    density_injection="bias-free X attention Q/K and X-to-Y attention K",
                    geometry_roundoff="float64 gaps; uniform if max gap deviation <= 2*input_eps*max(abs(time),mean_gap)",
                    x_variable_relation_rank=self.x_context.layers[0].node_dim,
                )
                if is_graph
                else {}
            ),
            **(
                dict(
                    density_method="Voronoi node density; version 1",
                    gap_node_encoding=False,
                    density_location=self.x_context.density_location,
                    density_attention_encoding=self.x_context.use_density_encoding
                    and not is_density_node,
                    density_node_encoding=self.x_context.use_density_encoding
                    and is_density_node,
                    node_input_fusion="concat(value_linear,variable_embedding,time_features,Linear_no_bias(1,d)(s)) -> LayerNorm(4d) -> attention Q/K/V Linear(4d,d); parallel residual Linear_no_bias(4d,d)"
                    if is_concat
                    else "concat(value_linear,variable_embedding,time_features,s) -> Linear(3d+1,d) -> SiLU -> Linear(d,d)"
                    if is_density_node
                    else "value_linear + variable_embedding + time_features",
                    sampling_geometry="per-sample per-X-variable unique observed times",
                    density="rho=2/(delta_left+delta_right); endpoints rho=1/delta (symmetric continuation)",
                    density_feature="s=log(mean_gap*rho); no clipping; singleton unknown encoded as zero",
                    density_bias="none"
                    if is_density_node
                    else "-softplus(gamma_raw_head)*(s_receiver-s_source)^2; gamma initialized to 1",
                    density_injection="node input only; no separate density logit, Q/K or X-to-Y injection"
                    if is_density_node
                    else "independent X attention logit only; no node, Q/K or X-to-Y injection",
                    regular_zero="s=0; identical to matched node-fusion no-density ablation, not old additive encoder"
                    if is_density_node
                    else "per-variable uniform sampling gives s=0 and density bias=0, including endpoints",
                    geometry_roundoff="float64 gaps; same input-resolution tolerance as legacy irgraph",
                    x_variable_relation_rank=self.x_context.layers[0].node_dim,
                )
                if is_voronoi
                else {}
            ),
            **(
                dict(
                    node_encoding="concat",
                    node_channel_dim=self.hidden,
                    node_input_dim=4 * self.hidden,
                    attention_hidden_dim=self.hidden,
                    density_encoder="Linear(1,d,bias=False); s=0 maps to zero vector",
                    pre_attention_fusion_mlp=False,
                    no_density_ablation="zero entire density block; identical 4d network and parameters",
                )
                if is_concat
                else {}
            ),
        )

    def _heads(self, x):
        return x.reshape(*x.shape[:-1], self.heads, self.dim).transpose(1, 2)

    def _clean(self, n):
        valid = n.valid.bool()
        return Observations(
            torch.where(valid, n.values, 0.0),
            torch.where(valid, n.times, 0.0),
            torch.where(valid, n.variables, 0).long(),
            valid,
        )

    def _encode(self, n):
        return (
            self.value_enc(n.values[..., None])
            + self.variable(n.variables)
            + torch.sin(self.time(n.times[..., None]))
        )

    def _basis(self, lag):
        return torch.softmax(
            -0.5 * ((lag[..., None] - self.centers) / self.lag_width).square(), -1
        )

    def _logits(self, q, k, times, ids, x):
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        if self.mode != "cross":
            if self.use_time_bias:
                p = self._heads(torch.sin(self.time(times[..., None])))
                s = self._heads(torch.sin(self.time(x.times[..., None])))
                logits = logits + self.time_scale[None, :, None, None] * (
                    p @ s.transpose(-1, -2)
                )
            if self.use_variable_bias:
                u, w = self._heads(self.u(ids)), self._heads(self.w(x.variables))
                logits = logits + self.var_scale[None, :, None, None] * (
                    u @ w.transpose(-1, -2)
                )
        return logits

    @staticmethod
    def _weights(logits, legal):
        score = logits.masked_fill(~legal[:, None], -torch.inf)
        score = torch.where(legal.any(-1)[:, None, :, None], score, 0.0)
        return score.softmax(-1) * legal[:, None]

    def encode_relations(self, history_y, history_x):
        """Historical pairs -> [B,H,V,K] lag distribution and support.

        Each observed Y attends to causal historical X. Its weighted lag basis
        is averaged by Y variable. No future X or query features enter here.
        """
        y, x = self._clean(history_y), self._clean(history_x)
        state = x.values.new_zeros(
            x.values.shape[0], self.heads, self.num_variables, len(self.centers)
        )
        count = x.values.new_zeros(x.values.shape[0], 1, self.num_variables, 1)
        k = self._heads(self.k(self._encode(x)))
        for start in range(0, y.values.shape[1], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            ys = Observations(
                y.values[:, sl], y.times[:, sl], y.variables[:, sl], y.valid[:, sl]
            )
            q = self._heads(self.q(self._encode(ys)))
            lag = ys.times[:, :, None] - x.times[:, None, :]
            legal = ys.valid[:, :, None] & x.valid[:, None, :] & (lag >= 0)
            weights = self._weights(
                self._logits(q, k, ys.times, ys.variables, x), legal
            )
            distribution = torch.einsum("bhqs,bqsk->bhqk", weights, self._basis(lag))
            index = ys.variables[:, None, :, None].expand(
                -1, self.heads, -1, len(self.centers)
            )
            state = state.scatter_add(2, index, distribution)
            count = count.scatter_add(
                2,
                ys.variables[:, None, :, None],
                legal.any(-1)[:, None, :, None].to(count.dtype),
            )
        return state / count.clamp_min(1), count > 0

    def forward(
        self,
        features,
        query_times,
        query_ids,
        query_valid,
        sources,
        history_y=None,
        history_x=None,
    ):
        """[B,Q,D] -> [B,Q,D]; query tokens never become message sources."""
        x = self._clean(sources)
        density_key = 0.0
        if self.x_encoder_type == "irgraph":
            encoded_x, density = self.x_context.encode_with_geometry(x)
            density_key = self.x_context.cross_density_key(density)
        else:
            encoded_x = (
                self.x_context(x) if self.x_context is not None else self._encode(x)
            )
        k, v = (
            self._heads(self.k(encoded_x) + density_key),
            self._heads(self.v(encoded_x)),
        )
        relation = support = None
        if self.mode == "v2" and self.transfer == "dynamic":
            if history_y is None or history_x is None:
                raise ValueError("V2 requires explicit historical observations")
            relation, support = self.encode_relations(history_y, history_x)
        output = []
        for start in range(0, features.shape[1], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            times, ids = query_times[:, sl], query_ids[:, sl]
            q = self._heads(self.q(self.query(features[:, sl])))
            logits = self._logits(q, k, times, ids, x)
            lag = times[:, :, None] - x.times[:, None, :]
            legal = query_valid[:, sl, None].bool() & x.valid[:, None, :]
            if not self.allow_future_x:
                legal = legal & (lag >= 0)
            if self.mode == "v2" and self.transfer != "none":
                if self.transfer == "static":
                    r = self.static_lag[ids].permute(0, 2, 1, 3).softmax(-1)
                    available = torch.ones_like(r[..., :1], dtype=torch.bool)
                else:
                    index = ids[:, None, :, None].expand(
                        -1, self.heads, -1, len(self.centers)
                    )
                    r = relation.gather(2, index)
                    available = support.expand(-1, self.heads, -1, -1).gather(
                        2, index[..., :1]
                    )
                bias = (
                    torch.einsum("bhqk,bqsk->bhqs", r, self._basis(lag))
                    .clamp_min(1e-8)
                    .log()
                )
                logits = logits + self.transfer_scale[
                    None, :, None, None
                ] * torch.where(available, bias, 0.0)
            weights = self._weights(logits, legal)
            message = (
                (weights @ v)
                .transpose(1, 2)
                .reshape(features.shape[0], -1, self.hidden)
            )
            delta = self.dropout(self.out(message))
            output.append(
                features[:, sl] + torch.where(legal.any(-1)[..., None], delta, 0.0)
            )
        return torch.cat(output, 1)
