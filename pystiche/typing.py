from typing import Union, Any, Sequence
import torch
from torch import nn

__all__ = [
    "TensorMeta",
    "ConvModule",
    "is_conv_module",
    "ConvModuleMeta",
    "PoolModule",
    "is_pool_module",
    "PoolModuleMeta",
]

Numeric = Union[int, float]

TensorMeta = Union[torch.device, torch.dtype]

ConvModule = Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]


def is_conv_module(x: Any) -> bool:
    return isinstance(x, (nn.Conv1d, nn.Conv2d, nn.Conv3d))


ConvModuleMeta = Union[int, Sequence[int]]

PoolModule = Union[
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d
]


def is_pool_module(x: Any) -> bool:
    return isinstance(
        x,
        (
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
        ),
    )


PoolModuleMeta = Union[int, Sequence[int]]
