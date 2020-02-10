from typing import Any, Sequence, Tuple
import torch
from pystiche.misc import verify_str_arg

__all__ = [
    "verify_is_image",
    "is_single_image",
    "is_batched_image",
    "is_image",
    "is_image_size",
    "is_edge_size",
    "calculate_aspect_ratio",
    "image_to_edge_size",
    "edge_to_image_size",
    "extract_batch_size",
    "extract_num_channels",
    "extract_image_size",
    "extract_edge_size",
    "extract_aspect_ratio",
]


def verify_is_image(x: Any):
    if not isinstance(x, torch.FloatTensor):
        msg = (
            f"pystiche uses torch.FloatTensor as native image type, but got input of "
            f"type {type(x)} instead."
        )
        raise TypeError(msg)

    if not x.dim() in (3, 4):
        msg = (
            f"pystiche uses CxHxW tensors for single and BxCxHxW tensors for batched "
            f"images, but got tensor with {x.dim()} dimensions instead."
        )
        raise TypeError(msg)


def is_single_image(x: Any) -> bool:
    return isinstance(x, torch.FloatTensor) and x.dim() == 3


def is_batched_image(x: Any) -> bool:
    return isinstance(x, torch.FloatTensor) and x.dim() == 4


def is_image(x: Any) -> bool:
    return is_single_image(x) or is_batched_image(x)


def is_image_size(x: Any) -> bool:
    try:
        return (
            isinstance(x, Sequence)
            and len(x) == 2
            and all(map(lambda item: isinstance(item, int), x))
        )
    except TypeError:
        return False


def is_edge_size(x: Any) -> bool:
    return isinstance(x, int)


def calculate_aspect_ratio(image_size: Tuple[int, int]) -> float:
    height, width = image_size
    return width / height


def image_to_edge_size(image_size: Tuple[int, int], edge: str = "short") -> int:
    edge = verify_str_arg(edge, "edge", ("short", "long", "vert", "horz"))
    if edge == "short":
        return min(image_size)
    elif edge == "long":
        return max(image_size)
    elif edge == "vert":
        return image_size[0]
    else:  # edge == "horz"
        return image_size[1]


def edge_to_image_size(
    edge_size: int, aspect_ratio: float, edge: str = "short"
) -> Tuple[int, int]:
    edge = verify_str_arg(edge, "edge", ("short", "long", "vert", "horz"))
    if edge == "vert":
        return edge_size, round(edge_size * aspect_ratio)
    elif edge == "horz":
        return round(edge_size * aspect_ratio), edge_size

    if (edge == "short") ^ (aspect_ratio < 1.0):
        return edge_size, round(edge_size * aspect_ratio)
    else:
        return round(edge_size / aspect_ratio), edge_size


def extract_batch_size(x: torch.Tensor) -> int:
    verify_is_image(x)
    if not is_batched_image(x):
        raise RuntimeError("Cannot extract a batch_size from a single image (CxHxW)")
    return x.size()[0]


def extract_num_channels(x: torch.Tensor) -> int:
    verify_is_image(x)
    return x.size()[-3]


def extract_image_size(x: torch.Tensor) -> Tuple[int, int]:
    verify_is_image(x)
    return tuple(x.size()[-2:])


def extract_edge_size(x: torch.Tensor, edge: str = "short") -> int:
    return image_to_edge_size(extract_image_size(x), edge=edge)


def extract_aspect_ratio(x: torch.Tensor) -> float:
    return calculate_aspect_ratio(extract_image_size(x))
