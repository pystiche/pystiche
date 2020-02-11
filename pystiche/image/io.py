from typing import Any, Union, Optional, Sequence
from PIL import Image
import torch
from .utils import is_image_size, is_edge_size, verify_is_single_image
from .transforms.functional import import_from_pil, export_to_pil
from .transforms import Resize, FixedAspectRatioResize


__all__ = ["read_image", "write_image", "show_image"]


def read_image(
    file: str,
    device: Union[torch.device, str] = "cpu",
    make_batched: bool = True,
    size: Optional[Union[int, Sequence[int]]] = None,
    **resize_kwargs: Any,
) -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)

    image = import_from_pil(Image.open(file), device=device, make_batched=make_batched)

    if size is None:
        return image

    if is_image_size(size):
        resize_transform = Resize(size, **resize_kwargs)
    elif is_edge_size(size):
        resize_transform = FixedAspectRatioResize(size, **resize_kwargs)
    else:
        raise ValueError
    return resize_transform(image)


def write_image(
    image: torch.Tensor, file: str, mode: Optional[str] = None, **kwargs: Any
):
    verify_is_single_image(image)
    export_to_pil(image, mode=mode).save(file, **kwargs)


def show_image(
    image: torch.Tensor, mode: Optional[str] = None, title: Optional[str] = None
):
    verify_is_single_image(image)
    export_to_pil(image, mode=mode).show(title=title)
