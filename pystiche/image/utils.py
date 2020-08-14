import functools
from typing import Any, Callable, Sequence, Tuple, cast

import torch

from pystiche.misc import verify_str_arg

__all__ = [
    "verify_is_single_image",
    "is_single_image",
    "verify_is_batched_image",
    "is_batched_image",
    "verify_is_image",
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
    "make_batched_image",
    "make_single_image",
    "force_image",
    "force_single_image",
    "force_batched_image",
]


def _verify_image_type(x: Any) -> None:
    msg = (
        "pystiche uses a torch.Tensor with dtype==torch.float32 "
        "as native image data type, but got input "
    )
    if not isinstance(x, torch.Tensor):
        msg += f"of type {type(x)} instead."
        raise TypeError(msg)
    elif x.dtype != torch.float32:
        msg += f"with dtype=={x.dtype} instead."
        raise TypeError(msg)


def _verify_single_image_dims(x: torch.Tensor) -> None:
    if x.dim() != 3:
        msg = (
            f"pystiche uses CxHxW tensors for single images, but got tensor with "
            f"{x.dim()} dimensions instead."
        )
        raise TypeError(msg)


def _verify_batched_image_dims(x: torch.Tensor) -> None:
    if x.dim() != 4:
        msg = (
            f"pystiche uses BxCxHxW tensors for batched images, but got tensor with "
            f"{x.dim()} dimensions instead."
        )
        raise TypeError(msg)


def _verify_image_dims(x: torch.Tensor) -> None:
    if x.dim() not in (3, 4):
        msg = (
            f"pystiche uses CxHxW tensors for single and BxCxHxW tensors for batched "
            f"images, but got tensor with {x.dim()} dimensions instead."
        )
        raise TypeError(msg)


def verify_is_single_image(x: Any) -> None:
    _verify_image_type(x)
    _verify_single_image_dims(x)


def is_single_image(x: Any) -> bool:
    try:
        verify_is_single_image(x)
    except TypeError:
        return False
    else:
        return True


def verify_is_batched_image(x: Any) -> None:
    _verify_image_type(x)
    _verify_batched_image_dims(x)


def is_batched_image(x: Any) -> bool:
    try:
        verify_is_batched_image(x)
    except TypeError:
        return False
    else:
        return True


def verify_is_image(x: Any) -> None:
    _verify_image_type(x)
    _verify_image_dims(x)


def is_image(x: Any) -> bool:
    try:
        verify_is_image(x)
    except TypeError:
        return False
    else:
        return True


def is_image_size(x: Any) -> bool:
    return (
        isinstance(x, Sequence)
        and len(x) == 2
        and all(map(lambda item: isinstance(item, int), x))
    )


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
        return round(edge_size / aspect_ratio), edge_size

    if (edge == "short") ^ (aspect_ratio < 1.0):
        return edge_size, round(edge_size * aspect_ratio)
    else:
        return round(edge_size / aspect_ratio), edge_size


def extract_batch_size(x: torch.Tensor) -> int:
    verify_is_batched_image(x)
    return x.size()[0]


def extract_num_channels(x: torch.Tensor) -> int:
    verify_is_image(x)
    return x.size()[-3]


def extract_image_size(x: torch.Tensor) -> Tuple[int, int]:
    verify_is_image(x)
    return cast(Tuple[int, int], tuple(x.size()[-2:]))


def extract_edge_size(x: torch.Tensor, edge: str = "short") -> int:
    return image_to_edge_size(extract_image_size(x), edge=edge)


def extract_aspect_ratio(x: torch.Tensor) -> float:
    return calculate_aspect_ratio(extract_image_size(x))


def make_batched_image(x: torch.Tensor) -> torch.Tensor:
    verify_is_single_image(x)
    return x.unsqueeze(0)


def make_single_image(x: torch.Tensor) -> torch.Tensor:
    batch_size = extract_batch_size(x)
    if batch_size != 1:
        msg = "ADDME"  # FIXME
        raise RuntimeError(msg)
    return x.squeeze(0)


def force_image(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        verify_is_image(x)
        return fn(x, *args, **kwargs)

    return wrapper


def force_single_image(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        verify_is_image(x)
        is_batched = is_batched_image(x)
        if is_batched:
            x = make_single_image(x)

        x = fn(x, *args, **kwargs)

        if is_batched and is_image(x):
            x = make_batched_image(x)

        return x

    return wrapper


def force_batched_image(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        verify_is_image(x)
        is_single = is_single_image(x)
        if is_single:
            x = make_batched_image(x)

        x = fn(x, *args, **kwargs)

        if is_single and is_image(x):
            x = make_single_image(x)

        return x

    return wrapper
