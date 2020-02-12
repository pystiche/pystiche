from typing import Union, Optional, Tuple
import torch
from torch.nn.functional import interpolate
from pystiche.misc import to_2d_arg
from pystiche.image.utils import (
    is_image_size,
    is_edge_size,
    edge_to_image_size,
    extract_image_size,
    extract_aspect_ratio,
    force_batched_image,
)
from ._utils import get_align_corners

__all__ = ["resize", "rescale"]


@force_batched_image
def resize(
    image: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    edge: str = "short",
    aspect_ratio: Optional[float] = None,
    interpolation_mode: str = "bilinear",
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = extract_aspect_ratio(image)

    if is_image_size(size):
        image_size = size
    elif is_edge_size(size):
        edge_size = size
        image_size = edge_to_image_size(edge_size, aspect_ratio, edge)
    else:
        raise RuntimeError

    return interpolate(
        image,
        size=image_size,
        scale_factor=None,
        mode=interpolation_mode,
        align_corners=get_align_corners(interpolation_mode),
    )


def rescale(
    image: torch.Tensor,
    factor: Union[float, Tuple[float, float]],
    interpolation_mode: str = "bilinear",
):
    height, width = extract_image_size(image)
    height_factor, width_factor = to_2d_arg(factor)
    height = round(height * height_factor)
    width = round(width * width_factor)
    image_size = (height, width)

    return resize(image, image_size, interpolation_mode=interpolation_mode)
