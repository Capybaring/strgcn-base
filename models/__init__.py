
from .STRGCN.STRGCN import STRGCN
from .STRGCN.ExEn import ExEn
from .STRGCN.STRGCNDirect import STRGCNDirect
from .STRGCN.ExEnAligned import ExEnAligned
from .STRGCN.ExEnGraphTransformer import ExEnGraphTransformer


def __getattr__(name):
    if name == 'STRGCNCLS':
        from .STRGCNCLS.STRGCNCLS import STRGCNCLS
        return STRGCNCLS
    # ExEn and non-aggregating STRGCN do not need GP dependencies.
    if name == 'GaussianBasedGraphAggregationMaskGenerator':
        from .STRGCN.GP_graph_aggr import GaussianBasedGraphAggregationMaskGenerator
        return GaussianBasedGraphAggregationMaskGenerator
    raise AttributeError(name)

__all__ = [
    "STRGCN",
    "ExEn",
    "STRGCNDirect",
    "ExEnAligned",
    "ExEnGraphTransformer",
    "STRGCNCLS",
    "GaussianBasedGraphAggregationMaskGenerator"
]
