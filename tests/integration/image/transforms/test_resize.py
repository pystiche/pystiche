import numpy as np
from PIL import Image

from pystiche import image as image_
from pystiche.image import transforms

from . import assert_transform_equals_pil


def test_Resize_image_size():
    def PILResizeTransform(image_size):
        size = image_size[::-1]
        return lambda image: image.resize(size, resample=Image.BILINEAR)

    image_size = (100, 100)
    pystiche_transform = transforms.Resize(image_size)
    pil_transform = PILResizeTransform(image_size)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform, mae=3e-2,
    )


def test_Resize_edge_size():
    def PILFixedAspectRatioResizeTransform(edge_size, edge):
        def transform(image):
            aspect_ratio = image_.calculate_aspect_ratio(image.size[::-1])
            image_size = image_.edge_to_image_size(edge_size, aspect_ratio, edge)
            size = image_size[::-1]
            return image.resize(size, resample=Image.BILINEAR)

        return transform

    edge_size = 100
    for edge in ("short", "long", "vert", "horz"):
        pystiche_transform = transforms.Resize(edge_size, edge=edge)
        pil_transform = PILFixedAspectRatioResizeTransform(edge_size, edge=edge)
        assert_transform_equals_pil(
            pystiche_transform=pystiche_transform,
            pil_transform=pil_transform,
            mae=3e-2,
        )


def test_Rescale():
    def PILRescaleTransform(factor):
        def transform(image):
            size = [round(edge_size * factor) for edge_size in image.size]
            return image.resize(size, resample=Image.BILINEAR)

        return transform

    factor = 1.0 / np.pi
    pystiche_transform = transforms.Rescale(factor)
    pil_transform = PILRescaleTransform(factor)
    assert_transform_equals_pil(
        pystiche_transform=pystiche_transform, pil_transform=pil_transform, mae=2e-2,
    )
