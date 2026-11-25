"""GraFITi, CRU and LatentODE adapters sharing the unchanged density plugin."""

import torch
from torch import nn

from baseline.irregular_backbones.cores import GraFITiCore, CRUCore, LatentODECore
from baseline.patch_fusion.adapter import DAGMLPFusion
from utils.tpatch_evaluation import variable_macro_loss
from .adapter import PatchQueryFusion
from .fusion import DynamicLagFusion


class IrregularQueryFusion(PatchQueryFusion):
    backbone_kind = None

    def __init__(
        self,
        global_args,
        hid_dim=64,
        latent_dim=20,
        n_heads=2,
        n_layer=4,
        ode_hidden=100,
        rec_layers=1,
        gen_layers=1,
        gru_units=100,
        num_basis=15,
        bandwidth=3,
        kl_weight=0.001,
        ode_rtol=1e-3,
        ode_atol=1e-4,
        fusion_mode="v1",
        fusion_hidden=64,
        plugin_hidden=32,
        plugin_heads=4,
        **fusion_params,
    ):
        nn.Module.__init__(self)
        if fusion_mode not in ("none", "mlp", "cross", "v1", "v2"):
            raise ValueError("Unsupported fusion mode")
        self.num_vars = global_args["num_of_vals"]
        self.target_ids = list(global_args["fusion_target_ids"])
        self.exogenous_ids = list(global_args["fusion_exogenous_ids"])
        self.backbone_ids = list(global_args["fusion_backbone_ids"])
        self.register_buffer(
            "target_column", torch.full((self.num_vars,), -1, dtype=torch.long)
        )
        self.target_column[self.target_ids] = torch.arange(len(self.target_ids))
        self.register_buffer("y_ids", torch.tensor(self.target_ids))
        self.mode, self.kl_weight = fusion_mode, kl_weight
        if kl_weight < 0:
            raise ValueError("kl_weight must be nonnegative")
        n = len(self.backbone_ids)
        if self.backbone_kind == "grafiti":
            self.core = GraFITiCore(n, hid_dim, n_layer, n_heads)
        elif self.backbone_kind == "cru":
            self.core = CRUCore(n, hid_dim, latent_dim, num_basis, bandwidth)
        else:
            self.core = LatentODECore(
                n,
                hid_dim,
                latent_dim,
                ode_hidden,
                rec_layers,
                gen_layers,
                gru_units,
                ode_rtol,
                ode_atol,
            )
        if fusion_mode == "mlp":
            self.fusion = DAGMLPFusion(
                len(self.target_ids), len(self.exogenous_ids), fusion_hidden
            )
        elif fusion_mode != "none":
            self.fusion = DynamicLagFusion(
                self.core.query_dim,
                self.num_vars,
                hidden=plugin_hidden,
                heads=plugin_heads,
                mode=fusion_mode,
                **fusion_params,
            )
        global_args["query_fusion_protocol"].update(
            mode=fusion_mode,
            backbone=self.backbone_kind,
            interface="official pre-output features; original prediction head",
            training_objective="variable-macro Gaussian NLL"
            if self.backbone_kind == "cru"
            else "variable-macro forecast MSE + kl_weight * KL(q(z0)||N(0,I))"
            if self.backbone_kind == "latentode"
            else "variable-macro forecast MSE",
            dense_collate="MTANQueryDataset is a history packing helper, not the backbone",
            latent_evaluation="posterior mean"
            if self.backbone_kind == "latentode"
            else "deterministic",
            cru_variant="full CRU; Van Loan covariance; float64 state; mean-head fusion only"
            if self.backbone_kind == "cru"
            else None,
        )
        if fusion_mode not in ("none", "mlp"):
            global_args["query_fusion_protocol"].update(self.fusion.protocol())

    def encode_queries(self, b):
        values = b["batch_backbone_mtan_values"]
        times = b["batch_backbone_mtan_times"]
        mask = b["batch_backbone_mtan_mask"].bool()
        queries = b["batch_backbone_queries"]
        features, variances, kls = [], [], []
        for i in range(len(values)):
            take = mask[i].any(-1)
            # Padded history rows must not change ODE/CRU evolution.
            if take.any():
                v, t, m = values[i, take], times[i, take], mask[i, take]
            else:
                v, t, m = values[i, :1] * 0, times[i, :1] * 0, mask[i, :1] & False
            query_nodes = b["batch_pred_mask"][i].bool() & b["batch_pad_mask"][i].bool()
            slots = b["batch_backbone_query_slot"][i, query_nodes]
            q = int(slots.max()) + 1 if len(slots) else 0
            f = self.core.query_features(v, t, m, queries[i, :q], self.training)
            if self.backbone_kind == "grafiti":
                f = f[:, : len(self.target_ids)]
            else:
                f = f[:, None].expand(-1, len(self.target_ids), -1)
            # The plugin keeps its own dtype; CRU filtering remains float64.
            dtype = (
                next(self.fusion.parameters()).dtype
                if self.mode not in ("none", "mlp")
                else f.dtype
            )
            f = f.to(dtype)
            features.append(
                torch.cat([f, f.new_zeros(queries.shape[1] - q, *f.shape[1:])])
            )
            if self.backbone_kind == "cru":
                variance = self.core.forecast_variance[:, : len(self.target_ids)]
                variances.append(
                    torch.cat(
                        [
                            variance,
                            variance.new_ones(queries.shape[1] - q, variance.shape[-1]),
                        ]
                    )
                )
            if self.backbone_kind == "latentode":
                kls.append(self.core.kl)
        self._forecast_variance = torch.stack(variances) if variances else None
        self._kl = torch.stack(kls).mean() if kls else values.new_zeros(())
        return torch.stack(features)

    def decode(self, features):
        head = self.core.output_layer
        features = features.to(head.weight.dtype)
        if self.backbone_kind == "grafiti":
            return head(features).squeeze(-1)
        if self.backbone_kind == "cru":
            for layer in self.core.decoder._hidden_layers_mean:
                features = layer(features)
        n = len(self.target_ids)
        return torch.einsum("bqyd,yd->bqy", features, head.weight[:n]) + head.bias[:n]

    def forecast_loss(self, prediction, b):
        if self.backbone_kind != "cru":
            return variable_macro_loss(
                prediction,
                b["batch_value"],
                b["batch_pred_mask"].bool() & b["batch_pad_mask"].bool(),
                b["batch_var_idx"],
                self.num_vars,
            )
        columns = self.target_column[b["batch_var_idx"].long()].clamp_min(0)
        variance = self._forecast_variance[
            torch.arange(len(prediction), device=prediction.device)[:, None],
            b["batch_backbone_query_slot"],
            columns,
        ].clamp_min(1e-8)
        valid = b["batch_pred_mask"].bool() & b["batch_pad_mask"].bool()
        losses = []
        for vid in self.target_ids:
            take = valid & (b["batch_var_idx"] == vid)
            if take.any():
                error = prediction[take] - b["batch_value"][take]
                losses.append(
                    0.5
                    * (
                        variance[take].log()
                        + error.square() / variance[take]
                        + 1.8378770664093453
                    ).mean()
                )
        return torch.stack(losses).mean() if losses else prediction.sum() * 0

    def auxiliary_loss(self):
        return self.kl_weight * self._kl


class GraFITiQueryFusion(IrregularQueryFusion):
    backbone_kind = "grafiti"


class CRUQueryFusion(IrregularQueryFusion):
    backbone_kind = "cru"


class LatentODEQueryFusion(IrregularQueryFusion):
    backbone_kind = "latentode"
