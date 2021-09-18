from os import path

import pyimagetest
import pytest

import torch

from pystiche import image as image_


class TestReadImage:
    def test_main(self, test_image_file, test_image):
        actual = image_.read_image(test_image_file)
        desired = test_image
        assert image_.is_batched_image(actual)
        pyimagetest.assert_images_almost_equal(actual, desired)

    def test_resize(self, test_image_file, test_image_pil):
        image_size = (200, 300)
        actual = image_.read_image(test_image_file, size=image_size)
        desired = test_image_pil.resize(image_size[::-1])
        pyimagetest.assert_images_almost_equal(actual, desired)

    def test_resize_scalar(self, test_image_file, test_image_pil):
        edge_size = 200

        aspect_ratio = image_.calculate_aspect_ratio(
            (test_image_pil.height, test_image_pil.width)
        )
        image_size = image_.edge_to_image_size(edge_size, aspect_ratio)

        actual = image_.read_image(test_image_file, size=edge_size)
        desired = test_image_pil.resize(image_size[::-1])
        pyimagetest.assert_images_almost_equal(actual, desired)

    def test_resize_other(self, test_image_file):
        with pytest.raises(TypeError):
            image_.read_image(test_image_file, size="invalid_size")


def test_write_image(tmpdir):
    torch.manual_seed(0)
    image = torch.rand(3, 100, 100)

    file = path.join(tmpdir, "tmp_image.png")
    image_.write_image(image, file)

    actual = image_.read_image(file=file)

    desired = image
    pyimagetest.assert_images_almost_equal(actual, desired)


def test_show_image_smoke(subtests, mocker, test_image_file, test_image):
    mocker.patch("pystiche.image.io._show_pil_image")
    image_.show_image(test_image)

    with subtests.test(image=test_image_file):
        image_.show_image(test_image_file)

    with subtests.test(image=None):
        with pytest.raises(TypeError):
            image_.show_image(None)

    with subtests.test(size=100):
        image_.show_image(test_image, size=100)

    with subtests.test(size=(100, 200)):
        image_.show_image(test_image, size=(100, 200))
