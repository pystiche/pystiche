from typing import Any, Union, Sequence, Dict
import torch
from pystiche.typing import (
    TensorMeta,
    ConvModule,
    ConvModuleMeta,
    PoolModule,
    PoolModuleMeta,
)
from .misc import prod, to_1d_arg, to_2d_arg, to_3d_arg, zip_equal


__all__ = [
    "tensor_meta",
    "conv_module_meta",
    "pool_module_meta",
    "extract_patches1d",
    "extract_patches2d",
    "extract_patches3d",
    "flatten_examplewise",
    "flatten_channelwise",
]


def _extract_meta_attrs(
    obj: Any, attrs: Sequence[str], **kwargs: Any
) -> Dict[str, Any]:
    for attr in attrs:
        if attr not in kwargs:
            kwargs[attr] = getattr(obj, attr)
    return kwargs


def tensor_meta(x: torch.Tensor, **kwargs: TensorMeta) -> Dict[str, TensorMeta]:
    attrs = ("dtype", "device")
    return _extract_meta_attrs(x, attrs, **kwargs)


def conv_module_meta(
    x: ConvModule, **kwargs: ConvModuleMeta
) -> Dict[str, ConvModuleMeta]:
    attrs = ("kernel_size", "stride", "padding", "dilation")
    return _extract_meta_attrs(x, attrs, **kwargs)


def pool_module_meta(
    x: PoolModule, **kwargs: PoolModuleMeta
) -> Dict[str, PoolModuleMeta]:
    attrs = ("kernel_size", "stride", "padding")
    return _extract_meta_attrs(x, attrs, **kwargs)


def _extract_patchesnd(
    x: torch.Tensor, patch_sizes: Sequence[int], strides: Sequence[int]
) -> torch.Tensor:
    num_channels = x.size()[1]
    dims = range(2, x.dim())
    for dim, patch_size, stride in zip_equal(dims, patch_sizes, strides):
        x = x.unfold(dim, patch_size, stride)
    x = x.permute(0, *dims, 1, *[dim + 2 for dim in dims]).contiguous()
    num_patches = prod(x.size()[: len(dims) + 1])
    return x.view(num_patches, num_channels, *patch_sizes)


def extract_patches1d(
    x: torch.Tensor,
    patch_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
) -> torch.Tensor:
    assert x.dim() == 3
    return _extract_patchesnd(x, to_1d_arg(patch_size), to_1d_arg(stride))


def extract_patches2d(
    x: torch.Tensor,
    patch_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
) -> torch.Tensor:
    assert x.dim() == 4
    return _extract_patchesnd(x, to_2d_arg(patch_size), to_2d_arg(stride))


def extract_patches3d(
    x: torch.Tensor,
    patch_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
) -> torch.Tensor:
    assert x.dim() == 5
    return _extract_patchesnd(x, to_3d_arg(patch_size), to_3d_arg(stride))


def flatten_examplewise(x: torch.Tensor) -> torch.Tensor:
    return torch.flatten(x, 1)


def flatten_channelwise(x: torch.Tensor) -> torch.Tensor:
    return torch.flatten(x, 2)
