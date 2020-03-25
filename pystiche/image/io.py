from typing import Any, Union, Optional, Tuple
from os import path, listdir
from PIL import Image
import torch
from torchvision.transforms.functional import (
    to_tensor as _to_tensor,
    to_pil_image as _to_pil_image,
)
from .utils import (
    extract_batch_size,
    is_batched_image,
    make_single_image,
    make_batched_image,
    force_single_image,
    force_image,
)
from .transforms.functional import resize  # FIXME: remove


__all__ = [
    "import_from_pil",
    "export_to_pil",
    "read_image",
    "read_guides",
    "write_image",
    "show_image",
]


def import_from_pil(
    image: Image.Image,
    device: Union[torch.device, str] = "cpu",
    make_batched: bool = True,
) -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)
    image = _to_tensor(image).to(device)
    if make_batched:
        image = make_batched_image(image)
    return image


@force_image
def export_to_pil(
    image: torch.Tensor, mode: Optional[str] = None
) -> Union[Image.Image, Tuple[Image.Image, ...]]:
    def fn(image: torch.Tensor) -> Image.Image:
        return _to_pil_image(image.detach().cpu().clamp(0.0, 1.0), mode)

    if is_batched_image(image):
        batched_image = image
        batch_size = extract_batch_size(batched_image)
        if batch_size == 1:
            return fn(make_single_image(batched_image))
        else:
            return tuple([fn(single_image) for single_image in batched_image])

    return fn(image)


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
