from typing import Union, Sequence
import torch
from torch import nn

__all__ = [
    "Numeric",
    "TensorMeta",
    "ConvModule",
    "ConvModuleMeta",
    "PoolModule",
    "PoolModuleMeta",
]

Numeric = Union[int, float]

TensorMeta = Union[torch.device, torch.dtype]

ConvModule = Union[nn.Conv1d, nn.Conv2d, nn.Conv2d]
ConvModuleMeta = Union[int, Sequence[int]]

PoolModule = Union[
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d
]
PoolModuleMeta = Union[int, Sequence[int]]
