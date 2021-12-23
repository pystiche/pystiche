from os import path
from typing import Any, Optional, Tuple, Union, cast

from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image as _to_pil_image
from torchvision.transforms.functional import to_tensor as _to_tensor

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

    def _show_pil_image(image: Image.Image, title: Optional[str] = None) -> None:
        fig, ax = plt.subplots()
        ax.axis("off")
        if title is not None:
            ax.set_title(title)

        cmap = "gray" if image.mode in ("L", "1") else None
        ax.imshow(image, cmap=cmap)
        plt.show()


except ImportError:

    def _show_pil_image(image: Image.Image, title: Optional[str] = None) -> None:
        image.show(title=title)


__all__ = [
    "import_from_pil",
    "export_to_pil",
    "read_image",
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
    image = cast(torch.Tensor, _to_tensor(image).to(device))
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
        size = cast(Tuple[int, int], size)
        height, width = size
    elif is_edge_size(size):
        size = cast(int, size)
        height, width = edge_to_image_size(
            size, calculate_aspect_ratio((image.height, image.width))
        )
    else:
        msg = (
            f"size can either be an edge size (int) or an image size "
            f"(Tuple[int, int]), but got {type(size)}."
        )
        raise TypeError(msg)

    return image.resize((width, height), resample=_PIL_RESAMPLE_MAP[interpolation_mode])


def read_image(
    file: str,
    device: Union[torch.device, str] = "cpu",
    make_batched: bool = True,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    interpolation_mode: str = "bilinear",
) -> torch.Tensor:
    """Read an image from file with :mod:`PIL.Image` and return it as
    :class:`~torch.Tensor` .

    Args:
        file: Path to image file to be read.
        device: Device that the image is transferred to. Defaults to CPU.
        make_batched: If ``True``, a fake batch dimension is added to the image.
        size: Optional size the image is resized to.
        interpolation_mode: Interpolation mode that is used to perform the optional
            resizing. Valid modes are ``"nearest"``, ``"bilinear"``, and ``"bicubic"``.
            Defaults to ``"bilinear"``.
    """
    if isinstance(device, str):
        device = torch.device(device)

    image = Image.open(path.expanduser(file))

    if size is not None:
        image = _pil_resize(image, size, interpolation_mode)

    return import_from_pil(image, device=device, make_batched=make_batched)


@force_single_image
def write_image(
    image: torch.Tensor, file: str, mode: Optional[str] = None, **save_kwargs: Any
) -> None:
    """Write a :class:`~torch.Tensor` image to a file with :mod:`PIL.Image` .

    Args:
        image: Image to be written.
        file: Path to image file.
        mode: Optional image mode. See the `Pillow documentation
            <https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes>`_
            for details.
        **save_kwargs: Other parameters that are passed to :meth:`PIL.Image.Image.save`
            .
    """
    export_to_pil(image, mode=mode).save(file, **save_kwargs)


def show_image(
    image: Union[torch.Tensor, str],
    title: Optional[str] = None,
    mode: Optional[str] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    interpolation_mode: str = "bilinear",
) -> None:
    """Show an image and optionally read it from file first.

    .. note::

        ``show_image`` uses :func:`matplotlib.pyplot.imshow` as primary means to show
        images. If that is not available the native :meth:`PIL.Image.Image.show` is
        used as a fallback.

    Args:
        image: Image to be shown. If ``str`` this is treated as a path to an image and
            is read by :func:`~pystiche.image.read_image` .
        title: Optional title of the image.
        mode: Optional image mode. See the `Pillow documentation
            <https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes>`_
            for details.
        size: Optional size the image is resized to.
        interpolation_mode: Interpolation mode that is used to perform the optional
            resizing. Valid modes are ``"nearest"``, ``"bilinear"``, and ``"bicubic"``.
            Defaults to ``"bilinear"``.
    """
    if isinstance(image, torch.Tensor):
        image = export_to_pil(image, mode=mode)
    elif isinstance(image, str):
        image = Image.open(path.expanduser(image))
    else:
        msg = f"image can either be torch.Tensor or str, but got {type(image)}."  # type: ignore[unreachable]
        raise TypeError(msg)

    if size is not None:
        image = _pil_resize(image, size, interpolation_mode)

    _show_pil_image(image, title=title)
