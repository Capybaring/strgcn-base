"""mTAN query adapter using the unchanged shared exogenous fusion algorithm."""

import torch
from torch import nn

from baseline.mtan.core import MTANCore
from baseline.patch_fusion.adapter import DAGMLPFusion

from .adapter import PatchQueryFusion
from .fusion import DynamicLagFusion


class MTANQueryFusion(PatchQueryFusion):
    def __init__(
        self,
        global_args,
        hid_dim=64,
        latent_dim=32,
        gen_hidden=50,
        te_dim=128,
        num_ref_points=64,
        n_heads=1,
        fusion_mode="v1",
        fusion_hidden=64,
        plugin_hidden=32,
        plugin_heads=4,
        kl_weight=0.001,
        **fusion_params,
    ):
        nn.Module.__init__(self)
        if fusion_mode not in ("none", "mlp", "cross", "v1", "v2"):
            raise ValueError("Unsupported mTAN fusion mode")
        if kl_weight < 0:
            raise ValueError("kl_weight must be nonnegative")
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
        self.core = MTANCore(
            len(self.backbone_ids),
            hid_dim,
            latent_dim,
            gen_hidden,
            te_dim,
            num_ref_points,
            n_heads,
            global_args.get("device", "cpu"),
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
            backbone="mTAN official variational encoder-decoder adapted to forecasting",
            interface="shared query latent broadcast to targets; original output-head row per target",
            training_objective="variable-macro forecast MSE + kl_weight * mean-reference KL(q||N(0,I))",
            kl_weight=kl_weight,
            training_latent="one reparameterized sample",
            evaluation_latent="posterior mean, not Monte Carlo predictive mean",
            reference_times="linspace(0,1); same normalized physical clock as observations",
            patching="collate metadata helper only; mTAN receives observations, not patches",
        )

    def encode_queries(self, b):
        z = self.core.encode(
            b["batch_backbone_mtan_values"],
            b["batch_backbone_mtan_times"],
            b["batch_backbone_mtan_mask"],
            self.training,
        )
        features = self.core.query_features(z, b["batch_backbone_queries"])
        return features[:, :, None, :].expand(-1, -1, len(self.target_ids), -1)

    def decode(self, features):
        # Original MLP head, with its existing output row for each Y. Taking the
        # diagonal avoids predicting every variable for every target query.
        head = self.core.decoder.z0_to_obs
        hidden = head[:-1](features)
        n = len(self.target_ids)
        return (
            torch.einsum("bqyd,yd->bqy", hidden, head[-1].weight[:n])
            + head[-1].bias[:n]
        )

    def auxiliary_loss(self):
        if self.core.kl is None:
            raise RuntimeError("mTAN must run forward before requesting its KL loss")
        return self.kl_weight * self.core.kl
