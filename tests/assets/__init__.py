import functools
from os import path
from urllib.parse import urljoin

import dill
from PIL import Image

from torchvision.transforms.functional import to_tensor

HERE = path.abspath(path.dirname(__file__))

__all__ = ["get_image_url", "get_image_file", "read_image", "load_asset"]


def get_image_file(name):
    return path.join(HERE, "image", f"{name}.png")


def get_image_url(name):
    return urljoin(
        "https://raw.githubusercontent.com/pmeier/pystiche/master/tests/assets/image/",
        f"{name}.png",
    )


Image.open = functools.lru_cache()(Image.open)


def read_image(name, pil=False):
    image = Image.open(get_image_file(name))
    if pil:
        return image

    return to_tensor(image).unsqueeze(0)


@functools.lru_cache()
def load_asset(type, name, ext=".asset"):
    file = path.join(HERE, type, path.splitext(name)[0] + ext)

    with open(file, "rb") as fh:
        return dill.load(fh)
