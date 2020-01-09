from typing import Any, Union, Optional, Sequence
from PIL import Image
import torch
from .utils import is_image_size, is_edge_size
from .transforms.functional import import_from_pil, export_to_pil
from .transforms import Resize, FixedAspectRatioResize


__all__ = ["read_image", "write_image", "show_image"]


def read_image(
    file: str,
    device=torch.device("cpu"),
    size: Optional[Union[int, Sequence[int]]] = None,
    **kwargs: Any
) -> torch.Tensor:
    image = import_from_pil(Image.open(file), device)

    if size is None:
        return image

    if is_image_size(size):
        resize_transform = Resize(size, **kwargs)
    elif is_edge_size(size):
        resize_transform = FixedAspectRatioResize(size, **kwargs)
    else:
        raise ValueError
    return resize_transform(image)


def write_image(
    image: torch.Tensor, file: str, mode: Optional[str] = None, **kwargs: Any
):
    export_to_pil(image, mode=mode).save(file, **kwargs)


def show_image(
    image: torch.Tensor, mode: Optional[str] = None, title: Optional[str] = None
):
    export_to_pil(image, mode=mode).show(title=title)
