"""Keep raw observation nodes for the plugin; isolate the backbone's Y history."""

from .TPatchFusionDataset import TPatchFusionDataset


class QueryFusionDataset(TPatchFusionDataset):
    def __init__(self, *args, backbone_history="y", **kwargs):
        if backbone_history not in ("y", "yx"):
            raise ValueError("backbone_history must be y or yx")
        self.backbone_history = backbone_history
        super().__init__(*args, **kwargs)
        self.global_args["query_fusion_protocol"] = dict(
            backbone_history=backbone_history,
            source="raw observed X nodes",
            mask="physical source_time <= query_time; queries never sources",
            time_unit="dataset normalized time; lag_max uses the same unit",
        )

    def long_collate_fn(self, batch):
        out = super().long_collate_fn(batch)
        if self.backbone_history == "y":
            n = len(self.packet["target_ids"])
            out["batch_backbone_mask"][..., n:] = 0
            out["batch_backbone_values"][..., n:] = 0
            out["batch_backbone_times"][..., n:] = 0
        return out
