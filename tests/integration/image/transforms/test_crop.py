import pyimagetest
import pytest

import torch

import pystiche.image as image_
from pystiche.image import transforms

from . import assert_is_identity_transform


def test_Crop(subtests, test_image):
    origin = (200, 300)
    size = (50, 30)

    spatial_slices = {
        ("top", "left"): [
            slice(origin[0], origin[0] + size[0]),
            slice(origin[1], origin[1] + size[1]),
        ],
        ("bottom", "left"): [
            slice(origin[0] - size[0], origin[0]),
            slice(origin[1], origin[1] + size[1]),
        ],
        ("top", "right"): [
            slice(origin[0], origin[0] + size[0]),
            slice(origin[1] - size[1], origin[1]),
        ],
        ("bottom", "right"): [
            slice(origin[0] - size[0], origin[0]),
            slice(origin[1] - size[1], origin[1]),
        ],
    }

    for (vert_anchor, horz_anchor), spatial_slice in spatial_slices.items():
        with subtests.test(vert_anchor=vert_anchor, horz_anchor=horz_anchor):
            transform = transforms.Crop(
                origin, size, vert_anchor=vert_anchor, horz_anchor=horz_anchor
            )
            actual = transform(test_image)
            desired = test_image[(slice(None), slice(None), *spatial_slice)]
            pyimagetest.assert_images_almost_equal(actual, desired)


def test_TopLeftCrop(test_image):
    size = 200

    transform = transforms.TopLeftCrop(size)
    actual = transform(test_image)
    desired = test_image[:, :, :size, :size]
    pyimagetest.assert_images_almost_equal(actual, desired)


def test_BottomLeftCrop(test_image):
    size = 200

    transform = transforms.BottomLeftCrop(size)
    actual = transform(test_image)
    desired = test_image[:, :, -size:, :size]
    pyimagetest.assert_images_almost_equal(actual, desired)


def test_TopRightCrop(test_image):
    size = 200

    transform = transforms.TopRightCrop(size)
    actual = transform(test_image)
    desired = test_image[:, :, :size, -size:]
    pyimagetest.assert_images_almost_equal(actual, desired)


def test_BottomRightCrop(test_image):
    size = 200

    transform = transforms.BottomRightCrop(size)
    actual = transform(test_image)
    desired = test_image[:, :, -size:, -size:]
    pyimagetest.assert_images_almost_equal(actual, desired)


def test_CenterCrop():
    image = torch.rand(1, 1, 100, 100)
    size = 50

    transform = transforms.CenterCrop(size)
    actual = transform(image)
    desired = image[:, :, size // 2 : -size // 2, size // 2 : -size // 2]
    pyimagetest.assert_images_almost_equal(actual, desired)


def test_ValidRandomCrop():
    def randint(range):
        return torch.randint(range + 1, ()).item()

    image_size = (100, 100)
    crop_size = (10, 20)
    image = torch.rand(1, 1, *image_size)

    image_height, image_width = image_size
    crop_height, crop_width = crop_size
    torch.manual_seed(0)
    vert_origin = randint(image_height - crop_height)
    horz_origin = randint(image_width - crop_width)

    torch.manual_seed(0)
    transform = transforms.ValidRandomCrop(crop_size)

    actual = transform(image)
    desired = image[
        :,
        :,
        vert_origin : vert_origin + crop_height,
        horz_origin : horz_origin + crop_width,
    ]
    pyimagetest.assert_images_almost_equal(actual, desired)


def test_ValidRandomCrop_identity(test_image):
    size = image_.extract_image_size(test_image)
    transform = transforms.ValidRandomCrop(size)
    assert_is_identity_transform(transform, test_image)


def test_ValidRandomCrop_too_large(test_image):
    size = image_.extract_edge_size(test_image, edge="long") * 2
    transform = transforms.ValidRandomCrop(size)

    with pytest.raises(RuntimeError):
        transform(test_image)
