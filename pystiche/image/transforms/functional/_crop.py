from typing import Tuple, Union, cast

import torch

from pystiche.image.utils import (
    extract_image_size,
    force_batched_image,
    is_edge_size,
    is_image_size,
)
from pystiche.misc import verify_str_arg

__all__ = [
    "crop",
    "top_left_crop",
    "bottom_left_crop",
    "top_right_crop",
    "bottom_right_crop",
    "center_crop",
]


def _parse_size(size: Union[Tuple[int, int], int]) -> Tuple[int, int]:
    if is_image_size(size):
        return cast(Tuple[int, int], size)
    elif is_edge_size(size):
        edge_size = cast(int, size)
        return edge_size, edge_size
    else:
        msg = (
            f"size can either be an edge size (int) or an image size "
            f"(Tuple[int, int]), but got {type(size)}."
        )
        raise TypeError(msg)


@force_batched_image
def crop(
    x: torch.Tensor,
    origin: Tuple[int, int],
    size: Union[Tuple[int, int], int],
    vert_anchor: str = "top",
    horz_anchor: str = "left",
) -> torch.Tensor:
    verify_str_arg(vert_anchor, "vert_anchor", ("top", "bottom"))
    verify_str_arg(horz_anchor, "horz_anchor", ("left", "right"))

    vert_origin, horz_origin = origin
    height, width = _parse_size(size)

    def create_vert_slice() -> slice:
        if vert_anchor == "top":
            return slice(vert_origin, vert_origin + height)
        else:  # vert_anchor == "bottom"
            return slice(vert_origin - height, vert_origin)

    def create_horz_slice() -> slice:
        if horz_anchor == "left":
            return slice(horz_origin, horz_origin + width)
        else:  # horz_anchor == "right"
            return slice(horz_origin - width, horz_origin)

    vert_slice = create_vert_slice()
    horz_slice = create_horz_slice()
    return x[:, :, vert_slice, horz_slice]


def top_left_crop(x: torch.Tensor, size: Union[Tuple[int, int], int]) -> torch.Tensor:
    origin = (0, 0)
    return cast(
        torch.Tensor, crop(x, origin, size, vert_anchor="top", horz_anchor="left")
    )


def bottom_left_crop(
    x: torch.Tensor, size: Union[Tuple[int, int], int]
) -> torch.Tensor:
    height, _ = extract_image_size(x)
    origin = (height, 0)
    return cast(
        torch.Tensor, crop(x, origin, size, vert_anchor="bottom", horz_anchor="left")
    )


def top_right_crop(x: torch.Tensor, size: Union[Tuple[int, int], int]) -> torch.Tensor:
    _, width = extract_image_size(x)
    origin = (0, width)
    return cast(
        torch.Tensor, crop(x, origin, size, vert_anchor="top", horz_anchor="right")
    )


def bottom_right_crop(
    x: torch.Tensor, size: Union[Tuple[int, int], int]
) -> torch.Tensor:
    origin = extract_image_size(x)
    return cast(
        torch.Tensor, crop(x, origin, size, vert_anchor="bottom", horz_anchor="right")
    )


def center_crop(x: torch.Tensor, size: Union[Tuple[int, int], int]) -> torch.Tensor:
    image_size = extract_image_size(x)
    size = _parse_size(size)
    vert_origin = (image_size[0] - size[0]) // 2
    horz_origin = (image_size[1] - size[1]) // 2
    return cast(torch.Tensor, crop(x, (vert_origin, horz_origin), size))
