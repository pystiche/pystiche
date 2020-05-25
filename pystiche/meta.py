from typing import Any, Dict, Sequence, Union

import torch
from torch import nn

__all__ = [
    "tensor_meta",
    "is_scalar_tensor",
    "ConvModule",
    "is_conv_module",
    "conv_module_meta",
    "pool_module_meta",
    "PoolModule",
    "is_pool_module",
]


def _extract_meta_attrs(
    obj: Any, attrs: Sequence[str], **kwargs: Any
) -> Dict[str, Any]:
    for attr in attrs:
        if attr not in kwargs:
            kwargs[attr] = getattr(obj, attr)
    return kwargs


def tensor_meta(x: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
    attrs = ("dtype", "device")
    return _extract_meta_attrs(x, attrs, **kwargs)


def is_scalar_tensor(x: torch.Tensor) -> bool:
    return x.dim() == 0


ConvModule = Union[
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose2d,
]


def is_conv_module(x: Any) -> bool:
    return isinstance(
        x,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    )


def conv_module_meta(x: ConvModule, **kwargs: Any) -> Dict[str, Any]:
    attrs = ("kernel_size", "stride", "padding", "dilation")
    return _extract_meta_attrs(x, attrs, **kwargs)


PoolModule = Union[
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
]


def is_pool_module(x: Any) -> bool:
    return isinstance(
        x,
        (
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
        ),
    )


def pool_module_meta(x: PoolModule, **kwargs: Any) -> Dict[str, Any]:
    attrs = ("kernel_size", "stride", "padding")
    return _extract_meta_attrs(x, attrs, **kwargs)
