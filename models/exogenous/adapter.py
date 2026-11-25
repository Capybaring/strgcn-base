"""Thin feature adapters retaining the original patch encoder and decoder."""

import torch

from baseline.patch_fusion.adapter import DAGMLPFusion, PatchFusion

from .fusion import DynamicLagFusion, Observations


class PatchQueryFusion(PatchFusion):
    def __init__(
        self,
        global_args,
        backbone,
        fusion_mode="v1",
        plugin_hidden=32,
        plugin_heads=4,
        lag_bins=8,
        lag_max=1.0,
        chunk_size=128,
        transfer="dynamic",
        use_time_bias=True,
        use_variable_bias=True,
        x_encoder="pointwise",
        allow_future_x=False,
        x_layers=1,
        x_depth=2,
        x_node_dim=16,
        use_gap_encoding=True,
        use_density_encoding=True,
        density_location="attention",
        node_encoding="mlp",
        **kwargs,
    ):
        super().__init__(global_args, backbone, **kwargs)
        del self.fusion
        if fusion_mode not in ("none", "mlp", "cross", "v1", "v2"):
            raise ValueError("Unsupported query fusion mode")
        self.mode = fusion_mode
        dim = self.core.decoder[0].in_features
        if fusion_mode == "mlp":
            self.fusion = DAGMLPFusion(
                len(self.target_ids),
                len(self.exogenous_ids),
                kwargs.get("fusion_hidden", 64),
            )
        elif fusion_mode != "none":
            self.fusion = DynamicLagFusion(
                dim,
                self.num_vars,
                plugin_hidden,
                plugin_heads,
                fusion_mode,
                lag_bins,
                lag_max,
                chunk_size,
                use_time_bias=use_time_bias,
                use_variable_bias=use_variable_bias,
                transfer=transfer,
                x_encoder=x_encoder,
                allow_future_x=allow_future_x,
                x_layers=x_layers,
                x_depth=x_depth,
                x_node_dim=x_node_dim,
                use_gap_encoding=use_gap_encoding,
                use_density_encoding=use_density_encoding,
                density_location=density_location,
                node_encoding=node_encoding,
            )
        self.register_buffer("y_ids", torch.tensor(self.target_ids))
        global_args["query_fusion_protocol"].update(
            mode=fusion_mode,
            transfer=transfer,
            interface="pre-decoder query features; original historical encoder and decoder",
            lag_bins=lag_bins,
            lag_max=lag_max,
        )
        if fusion_mode not in ("none", "mlp"):
            global_args["query_fusion_protocol"].update(self.fusion.protocol())

    def encode_queries(self, b):
        return self.core.encode_queries(
            b["batch_backbone_queries"],
            b["batch_backbone_values"],
            b["batch_backbone_times"],
            b["batch_backbone_mask"],
        )[:, : len(self.target_ids)].transpose(1, 2)

    def decode(self, features):
        return self.core.decoder(features).squeeze(-1)

    @staticmethod
    def _observations(b, valid):
        indices = [torch.where(row)[0] for row in valid]
        width = max(1, max(map(len, indices)))
        shape = (valid.shape[0], width)
        values = b["batch_value"].new_zeros(shape)
        times = b["batch_timestamp"].new_zeros(shape)
        ids = b["batch_var_idx"].new_zeros(shape).long()
        mask = torch.zeros(shape, dtype=torch.bool, device=valid.device)
        for j, ix in enumerate(indices):
            n = len(ix)
            values[j, :n] = b["batch_value"][j, ix]
            times[j, :n] = b["batch_timestamp"][j, ix]
            ids[j, :n] = b["batch_var_idx"][j, ix]
            mask[j, :n] = True
        return Observations(values, times, ids, mask)

    def fuse(self, features, b):
        B, Q, Y, D = features.shape
        role, valid = b["batch_node_type"], b["batch_pad_mask"].bool()
        times = b["batch_backbone_queries"][..., None].expand(B, Q, Y).reshape(B, -1)
        ids = self.y_ids[None, None].expand(B, Q, Y).reshape(B, -1)
        source = valid & ((role == 0) | (role == 2))
        return self.fusion(
            features.reshape(B, -1, D),
            times,
            ids,
            torch.ones_like(times, dtype=torch.bool),
            self._observations(b, source),
            self._observations(b, valid & (role == 1)),
            self._observations(b, valid & (role == 0)),
        ).reshape(B, Q, Y, D)

    def forward(self, **b):
        features = self.encode_queries(b)
        if self.mode not in ("none", "mlp"):
            features = self.fuse(features, b)
        prediction = self.decode(features)
        if self.mode == "mlp":
            prediction = self.fusion(b["batch_backbone_future_x"], prediction)
        column = self.target_column[b["batch_var_idx"].long()].clamp_min(0)
        output = prediction[
            torch.arange(prediction.shape[0], device=prediction.device)[:, None],
            b["batch_backbone_query_slot"],
            column,
        ]
        return torch.where(
            b["batch_pred_mask"].bool() & b["batch_pad_mask"].bool(), output, 0.0
        ).unsqueeze(-1)


class TPatchQueryFusion(PatchQueryFusion):
    def __init__(self, global_args, **kwargs):
        super().__init__(global_args, "tpatchgnn", **kwargs)


class HiPatchQueryFusion(PatchQueryFusion):
    def __init__(self, global_args, **kwargs):
        super().__init__(global_args, "hipatch", **kwargs)
