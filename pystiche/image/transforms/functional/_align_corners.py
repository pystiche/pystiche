from typing import List, Optional, Tuple, Union, cast

import torch
from torch.nn.functional import affine_grid as _affine_grid
from torch.nn.functional import grid_sample as _grid_sample
from torch.nn.functional import interpolate as _interpolate

__all__ = ["interpolate", "affine_grid", "grid_sample"]

# The purpose of this module is to suppress UserWarnings from interpolate(),
# grid_sample(), and affine_grid() due to the changed default value of grid_sample


def _get_align_corners(interpolation_mode: str) -> Optional[bool]:
    if interpolation_mode in ("nearest", "area"):
        return None
    else:
        return False


def interpolate(
    image: torch.Tensor,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        _interpolate(
            image,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=_get_align_corners(mode),
        ),
    )


def affine_grid(theta: torch.Tensor, size: Tuple[int, int, int, int],) -> torch.Tensor:
    return _affine_grid(theta, cast(List[int], size), align_corners=False)


def grid_sample(
    image: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    return _grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=_get_align_corners(mode),
    )
