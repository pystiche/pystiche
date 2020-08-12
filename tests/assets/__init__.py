import functools
import os
from os import path
from urllib.parse import urljoin

from pystiche.image import read_image as _read_image

HERE = path.abspath(path.dirname(__file__))

__all__ = ["get_image_url", "get_image_file", "read_image"]


def get_image_file(name):
    return path.join(HERE, "image", f"{name}.png")


def get_image_url(name):
    return urljoin(
        "https://raw.githubusercontent.com/pmeier/pystiche/master/tests/assets/image/",
        f"{name}.png",
    )


# Since a torch.Tensor is mutable we only cache the raw input and clone the image for
# every call
_read_image = functools.lru_cache()(_read_image)


def read_image(name):
    image = _read_image(get_image_file(name))
    return image.clone()
