from .QueryFusionDataset import QueryFusionDataset
from .MTANQueryDataset import MTANQueryDataset
from .AllVariableForecastDataset import AllVariablePatchDataset, AllVariableMTANDataset
from .TPatchFusionDataset import TPatchFusionDataset
from .TPatchRecordDataset import TPatchRecordDataset
from .ExEnProcessedDataset import ExEnProcessedDataset
from .abs import AbstractDataset

from .ITSPredictionDataset import ITSPredictionDataset
from .ExEnPredictionDataset import ExEnPredictionDataset


def __getattr__(name):
    if name == 'ITSClassificationDataset':
        from .ITSClassificationDataset import ITSClassificationDataset
        return ITSClassificationDataset
    raise AttributeError(name)
__all__ = [
    "MTANQueryDataset",
    "TPatchFusionDataset",
    "TPatchRecordDataset",
    "ExEnProcessedDataset",
    "AbstractDataset",
    "ITSPredictionDataset",
    "ExEnPredictionDataset",
    "ITSClassificationDataset"
]
