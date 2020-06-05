from typing import Optional, Tuple, Union, cast

import torch

from pystiche.image.utils import (
    edge_to_image_size,
    extract_aspect_ratio,
    extract_image_size,
    force_batched_image,
    is_edge_size,
    is_image_size,
)
from pystiche.misc import to_2d_arg

from ._align_corners import interpolate

__all__ = ["resize", "rescale"]


@force_batched_image
def resize(
    image: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    edge: str = "short",
    aspect_ratio: Optional[float] = None,
    interpolation_mode: str = "bilinear",
) -> torch.Tensor:
    r"""Resize an image as specified. See :class:`~pystiche.image.transforms.Resize`
    for details.
    """
    if aspect_ratio is None:
        aspect_ratio = extract_aspect_ratio(image)

    if is_image_size(size):
        image_size = size
    elif is_edge_size(size):
        image_size = edge_to_image_size(cast(int, size), aspect_ratio, edge)
    else:
        msg = (
            f"size can either be an edge size (int) or an image size "
            f"(Tuple[int, int]), but got {type(size)}."
        )
        raise TypeError(msg)

    return interpolate(
        image, size=image_size, scale_factor=None, mode=interpolation_mode,
    )


def rescale(
    image: torch.Tensor,
    factor: Union[float, Tuple[float, float]],
    interpolation_mode: str = "bilinear",
) -> torch.Tensor:
    height, width = extract_image_size(image)
    height_factor, width_factor = to_2d_arg(factor)
    height = round(height * height_factor)
    width = round(width * width_factor)
    image_size = (height, width)

    return cast(
        torch.Tensor, resize(image, image_size, interpolation_mode=interpolation_mode)
    )
