import warnings
from typing import Sequence, Union

import torch

from pystiche.misc import prod, to_1d_arg, to_2d_arg, to_3d_arg, zip_equal

__all__ = [
    "extract_patches1d",
    "extract_patches2d",
    "extract_patches3d",
]


def _warn_output_shape(*dim_names: str) -> None:
    name = f"extract_patches{len(dim_names)}d"
    partial_shape = "x".join(dim_names)
    warnings.warn(
        f"The output shape of {name} will change in the future. "
        f"The current shape B*PxCx{partial_shape} will be replaced by "
        f"BxPxCx{partial_shape} thus adding a dimension. "
        f"Here, 'P' denotes the number of extracted patches.",
        FutureWarning,
    )


def _extract_patchesnd(
    x: torch.Tensor, patch_sizes: Sequence[int], strides: Sequence[int]
) -> torch.Tensor:
    num_channels = x.size()[1]
    dims = range(2, x.dim())
    for dim, patch_size, stride in zip_equal(dims, patch_sizes, strides):
        x = x.unfold(dim, patch_size, stride)
    x = x.permute(0, *dims, 1, *[dim + len(dims) for dim in dims]).contiguous()
    num_patches = prod(x.size()[: len(dims) + 1])
    return x.view(num_patches, num_channels, *patch_sizes)


def extract_patches1d(
    x: torch.Tensor,
    patch_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
) -> torch.Tensor:
    assert x.dim() == 3
    _warn_output_shape("L")
    return _extract_patchesnd(x, to_1d_arg(patch_size), to_1d_arg(stride))


def extract_patches2d(
    x: torch.Tensor,
    patch_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
) -> torch.Tensor:
    assert x.dim() == 4
    _warn_output_shape("H", "W")
    return _extract_patchesnd(x, to_2d_arg(patch_size), to_2d_arg(stride))


def extract_patches3d(
    x: torch.Tensor,
    patch_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
) -> torch.Tensor:
    assert x.dim() == 5
    _warn_output_shape("D", "H", "W")
    return _extract_patchesnd(x, to_3d_arg(patch_size), to_3d_arg(stride))
