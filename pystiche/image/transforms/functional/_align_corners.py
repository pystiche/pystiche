from typing import Union, Optional, Tuple
import torch
from torch.nn.functional import (
    interpolate as _interpolate,
    affine_grid as _affine_grid,
    grid_sample as _grid_sample,
)

__all__ = ["interpolate", "affine_grid", "grid_sample"]

# The purpose of this module is to suppress UserWarnings from interpolate(),
# grid_sample(), and affine_grid() due to the changed default value of align_corners.


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
    return _interpolate(
        image,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=_get_align_corners(mode),
    )


def affine_grid(theta: torch.Tensor, size: Tuple[int, int, int, int],) -> torch.Tensor:
    # I have no idea why mypy thinks that align_corners is an unexpected keyword
    # argument, when it clearly is one.
    return _affine_grid(theta, list(size), align_corners=False)  # type: ignore[call-arg]


def grid_sample(
    image: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    # See comment above comment in affine_grid().
    return _grid_sample(  # type: ignore[call-arg]
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=_get_align_corners(mode),
    )
