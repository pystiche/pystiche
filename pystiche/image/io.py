import warnings
from os import listdir, path
from typing import Any, Optional, Tuple, Union

from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image as _to_pil_image
from torchvision.transforms.functional import to_tensor as _to_tensor

from pystiche.misc import build_deprecation_message

from .utils import (
    calculate_aspect_ratio,
    edge_to_image_size,
    extract_batch_size,
    force_image,
    force_single_image,
    is_batched_image,
    is_edge_size,
    is_image_size,
    make_batched_image,
    make_single_image,
)

try:
    import matplotlib.pyplot as plt

    def _show_pil_image(image: Image.Image) -> None:
        plt.imshow(image)


except ImportError:

    def _show_pil_image(image: Image.Image) -> None:
        image.show()


__all__ = [
    "import_from_pil",
    "export_to_pil",
    "read_image",
    "read_guides",
    "write_image",
    "show_image",
]

_PIL_RESAMPLE_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


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


def _pil_resize(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
    interpolation_mode: str,
    **kwargs: Any,
) -> Image.Image:
    if is_image_size(size):
        height, width = size
    elif is_edge_size(size):
        height, width = edge_to_image_size(
            size, calculate_aspect_ratio((image.height, image.width))
        )
    else:
        raise RuntimeError

    if kwargs:
        msg = build_deprecation_message(
            "Passing additional parameters via **resize_kwargs",
            "0.4.0",
            info="The keyword arguments are ignored.",
        )
        warnings.warn(msg)

    return image.resize((width, height), resample=_PIL_RESAMPLE_MAP[interpolation_mode])


def read_image(
    file: str,
    device: Union[torch.device, str] = "cpu",
    make_batched: bool = True,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    interpolation_mode: str = "bilinear",
    **resize_kwargs: Any,
) -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)

    image = Image.open(file)

    if size is not None:
        image = _pil_resize(image, size, interpolation_mode, **resize_kwargs)

    return import_from_pil(image, device=device, make_batched=make_batched)


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
    interpolation_mode: str = "bilinear",
    **resize_kwargs: Any,
):
    image = export_to_pil(image, mode=mode)

    if size is not None:
        image = _pil_resize(image, size, interpolation_mode, **resize_kwargs)

    _show_pil_image(image)
