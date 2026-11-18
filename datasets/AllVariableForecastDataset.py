"""All-variable native backbones; endogenous IDs affect evaluation only."""

from .MTANQueryDataset import MTANQueryDataset
from .QueryFusionDataset import QueryFusionDataset


class AllVariableForecastMixin:
    def __init__(self, data, type, global_args, **kwargs):
        packet = data.get(f"{type}_data", data)
        # Copy metadata, never change the processor's packets or raw records.
        all_ids = list(range(len(packet["names"])))
        packet = dict(
            packet,
            evaluation_target_ids=list(packet["target_ids"]),
            role_exogenous_ids=list(packet["exogenous_ids"]),
            target_ids=all_ids,
            exogenous_ids=[],
            baseline_ids=[],
        )
        kwargs.update(
            backbone_history="yx", known_future_exogenous_var_ids=[], use_exogenous=True
        )
        super().__init__(packet, type=type, global_args=global_args, **kwargs)
        if "tpatch_protocol" in global_args:
            global_args["tpatch_protocol"] = dict(
                global_args["tpatch_protocol"],
                condition="all_variable_history_only_no_known_future_values",
                training_excluded_target_ids=[],
            )
        global_args["original_baseline_protocol"] = dict(
            input="all original variable histories; no future values",
            training_target_ids=all_ids,
            evaluation_target_ids=packet["evaluation_target_ids"],
            role_exogenous_ids=packet["role_exogenous_ids"],
            fusion="none",
            static_handling="original valid observations, no synthetic future baseline copies",
            windows="same record splits and Y-eligible windows as plugin comparisons",
        )


class AllVariablePatchDataset(AllVariableForecastMixin, QueryFusionDataset):
    pass


class AllVariableMTANDataset(AllVariableForecastMixin, MTANQueryDataset):
    pass
