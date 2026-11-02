from .abs import AbstractDataset

from .ITSPredictionDataset import ITSPredictionDataset
from .ExEnPredictionDataset import ExEnPredictionDataset


def __getattr__(name):
    if name == 'ITSClassificationDataset':
        from .ITSClassificationDataset import ITSClassificationDataset
        return ITSClassificationDataset
    raise AttributeError(name)
__all__ = [
    "AbstractDataset",
    "ITSPredictionDataset",
    "ExEnPredictionDataset",
    "ITSClassificationDataset"
]
