from typing import Any, Union, Optional, Tuple
from os import path, listdir
from PIL import Image
import torch
from .utils import force_single_image
from .transforms.functional import import_from_pil, export_to_pil, resize


__all__ = ["read_image", "write_image", "show_image"]


def read_image(
    file: str,
    device: Union[torch.device, str] = "cpu",
    make_batched: bool = True,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    **resize_kwargs: Any,
) -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)

    image = import_from_pil(Image.open(file), device=device, make_batched=make_batched)

    if size is not None:
        image = resize(image, size, **resize_kwargs)

    return image


def read_guides(
    dir: str, device: Union[torch.device, str] = "cpu", make_batched: bool = True,
):
    def import_image(file):
        image = Image.open(path.join(dir, file)).convert("1")
        return import_from_pil(image, device=device, make_batched=make_batched)

    return {path.splitext(file)[0]: import_image(file) for file in listdir(dir)}


@force_single_image
def write_image(
    image: torch.Tensor, file: str, mode: Optional[str] = None, **kwargs: Any
):
    export_to_pil(image, mode=mode).save(file, **kwargs)


@force_single_image
def show_image(
    image: torch.Tensor,
    mode: Optional[str] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    **resize_kwargs: Any,
):
    if size is not None:
        image = resize(image, size, **resize_kwargs)
    export_to_pil(image, mode=mode).show()
