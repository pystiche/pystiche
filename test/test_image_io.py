from os import path
import unittest
import torch
from pystiche.image import io
from image_testcase import PysticheImageTestcase
from utils import get_tmp_dir


class Tester(PysticheImageTestcase, unittest.TestCase):
    def test_read_image(self):
        actual = io.read_image(self.default_test_image_file)
        desired = self.load_image("pystiche")
        self.assertImagesAlmostEqual(actual, desired)

    def test_write_image(self):
        torch.manual_seed(0)
        image = torch.rand(3, 100, 100)
        with get_tmp_dir() as tmp_dir:
            file = path.join(tmp_dir, "tmp_image.png")
            io.write_image(image, file)

            actual = self.load_image("pystiche", file=file)

        desired = image
        self.assertImagesAlmostEqual(actual, desired)

    def test_show_image(self):
        pass
