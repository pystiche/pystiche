import math
from os import path

import torch

from pystiche import misc

from .utils import PysticheTestCase, get_tmp_dir


class TestMisc(PysticheTestCase):
    def test_prod(self):
        n = 10
        iterable = range(1, n + 1)

        actual = misc.prod(iterable)
        desired = math.factorial(n)
        self.assertEqual(actual, desired)

    def test_to_1d_arg(self):
        val = 0
        actual = misc.to_1d_arg(val)
        desired = (val,)
        self.assertTupleEqual(actual, desired)

        val = (0,)
        actual = misc.to_1d_arg(val)
        desired = val
        self.assertTupleEqual(actual, desired)

        val = 0
        actual = misc.to_1d_arg([val])
        desired = (val,)
        self.assertTupleEqual(actual, desired)

        val = (0, 0)
        with self.assertRaises(RuntimeError):
            misc.to_1d_arg(val)

    def test_to_2d_arg(self):
        val = 0
        actual = misc.to_2d_arg(val)
        desired = (val, val)
        self.assertTupleEqual(actual, desired)

        val = (0, 0)
        actual = misc.to_2d_arg(val)
        desired = val
        self.assertTupleEqual(actual, desired)

        val = 0
        actual = misc.to_2d_arg([val] * 2)
        desired = (val, val)
        self.assertTupleEqual(actual, desired)

        val = (0,)
        with self.assertRaises(RuntimeError):
            misc.to_2d_arg(val)

    def test_to_3d_arg(self):
        val = 0
        actual = misc.to_3d_arg(val)
        desired = (val, val, val)
        self.assertTupleEqual(actual, desired)

        val = (0, 0, 0)
        actual = misc.to_3d_arg(val)
        desired = val
        self.assertTupleEqual(actual, desired)

        val = 0
        actual = misc.to_3d_arg([val] * 3)
        desired = (val, val, val)
        self.assertTupleEqual(actual, desired)

        val = (0,)
        with self.assertRaises(RuntimeError):
            misc.to_3d_arg(val)

    def test_zip_equal(self):
        foo = (1, 2)
        bar = ("a", "b")

        actual = tuple(misc.zip_equal(foo, bar))
        desired = tuple(zip(foo, bar))
        self.assertTupleEqual(actual, desired)

        foo = (1, 2)
        bar = ("a", "b", "c")

        with self.assertRaises(RuntimeError):
            misc.zip_equal(foo, bar)

    def test_verify_str_arg(self):
        arg = None
        with self.assertRaises(ValueError):
            misc.verify_str_arg(arg)

        arg = "foo"
        valid_args = ("bar", "baz")
        with self.assertRaises(ValueError):
            misc.verify_str_arg(arg, valid_args=valid_args)

        arg = "foo"
        valid_args = ("foo", "bar")

        actual = misc.verify_str_arg(arg, valid_args=valid_args)
        desired = arg
        self.assertEqual(actual, desired)

    def assertScalarTensorAlmostEqual(self, actual, desired):
        self.assertAlmostEqual(actual.item(), desired.item())

    def test_get_input_image_tensor(self):
        image = torch.tensor(0.0)

        starting_point = image
        actual = misc.get_input_image(starting_point)
        desired = image
        self.assertIsNot(actual, desired)
        self.assertScalarTensorAlmostEqual(actual, desired)

    def test_get_input_image_tensor_content(self):
        starting_point = "content"
        image = torch.tensor(0.0)

        actual = misc.get_input_image(starting_point, content_image=image)
        desired = image
        self.assertScalarTensorAlmostEqual(actual, desired)

        with self.assertRaises(RuntimeError):
            misc.get_input_image(starting_point, style_image=image)

    def test_get_input_image_tensor_style(self):
        starting_point = "style"
        image = torch.tensor(0.0)

        actual = misc.get_input_image(starting_point, style_image=image)
        desired = image
        self.assertScalarTensorAlmostEqual(actual, desired)

        with self.assertRaises(RuntimeError):
            misc.get_input_image(starting_point, content_image=image)

    def test_get_input_image_tensor_random(self):
        def assertTensorDtypeEqual(actual, desired):
            self.assertEqual(actual.dtype, desired.dtype)

        starting_point = "random"
        content_image = torch.tensor(0.0, dtype=torch.float32)
        style_image = torch.tensor(0.0, dtype=torch.float64)

        actual = misc.get_input_image(starting_point, content_image=content_image)
        desired = content_image
        assertTensorDtypeEqual(actual, desired)

        actual = misc.get_input_image(starting_point, style_image=style_image)
        desired = style_image
        assertTensorDtypeEqual(actual, desired)

        actual = misc.get_input_image(
            starting_point, content_image=content_image, style_image=style_image
        )
        desired = content_image
        assertTensorDtypeEqual(actual, desired)

        with self.assertRaises(RuntimeError):
            misc.get_input_image(starting_point)

    def test_get_device(self):
        actual = misc.get_device()
        desired = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(actual, desired)

    def test_get_device_str(self):
        device_name = "mkldnn"
        actual = misc.get_device(device_name)
        desired = torch.device(device_name)
        self.assertEqual(actual, desired)

    def test_download_file(self):
        with get_tmp_dir() as root:
            url = "https://raw.githubusercontent.com/pmeier/pystiche/master/tests/assets/image/test_image.png"
            file = path.join(root, path.basename(url))
            misc.download_file(url, file)

            actual = self.load_image(file=file)
            desired = self.load_image()
            self.assertImagesAlmostEqual(actual, desired)

    def test_reduce(self):
        torch.manual_seed(0)
        x = torch.rand(1, 3, 128, 128)

        actual = misc.reduce(x, "mean")
        desired = torch.mean(x)
        self.assertTensorAlmostEqual(actual, desired)

        actual = misc.reduce(x, "sum")
        desired = torch.sum(x)
        self.assertTensorAlmostEqual(actual, desired)

        actual = misc.reduce(x, "none")
        desired = x
        self.assertTensorAlmostEqual(actual, desired)
