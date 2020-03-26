from os import path
import unittest
import torch
from pystiche.image import io, calculate_aspect_ratio, edge_to_image_size
from image_test_case import PysticheImageTestCase
from utils import get_tmp_dir


class Tester(PysticheImageTestCase, unittest.TestCase):
    def test_read_image(self):
        actual = io.read_image(self.default_image_file())
        desired = self.load_image()
        self.assertImagesAlmostEqual(actual, desired)

    def test_read_image_resize(self):
        image_size = (200, 300)
        actual = io.read_image(self.default_image_file(), size=image_size)
        desired = self.load_image(backend="PIL").resize(image_size[::-1])
        self.assertImagesAlmostEqual(actual, desired)

    def test_read_image_resize_scalar(self):
        edge_size = 200

        image = self.load_image(backend="PIL")
        aspect_ratio = calculate_aspect_ratio((image.height, image.width))
        image_size = edge_to_image_size(edge_size, aspect_ratio)

        actual = io.read_image(self.default_image_file(), size=edge_size)
        desired = image.resize(image_size[::-1])
        self.assertImagesAlmostEqual(actual, desired)

    def test_write_image(self):
        torch.manual_seed(0)
        image = torch.rand(3, 100, 100)
        with get_tmp_dir() as tmp_dir:
            file = path.join(tmp_dir, "tmp_image.png")
            io.write_image(image, file)

            actual = self.load_image(file=file)

        desired = image
        self.assertImagesAlmostEqual(actual, desired)

    def test_show_image(self):
        pass
