import math
import unittest
from os import path

import torch

from pystiche.misc import cuda, misc

from .utils import PysticheTestCase, get_tmp_dir, skip_if_cuda_not_available


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

    def test_to_eng(self):
        def assertEqual(actual, desired):
            self.assertAlmostEqual(actual[0], desired[0])
            self.assertEqual(actual[1], desired[1])

        val = 123456789
        actual = misc.to_eng(val)
        desired = (123.456789, 6)
        assertEqual(actual, desired)

        val = -9876e-5
        actual = misc.to_eng(val)
        desired = (-98.76, -3)
        assertEqual(actual, desired)

    def test_to_eng_zero(self):
        eps = 1e-8
        vals = (0.0, 1e-1 * eps, -1e-1 * eps)

        for val in vals:
            sig, exp = misc.to_eng(val, eps=eps)
            self.assertAlmostEqual(sig, 0.0)
            self.assertEqual(exp, 0)

    def test_to_engstr(self):
        val = 123456789
        actual = misc.to_engstr(val)
        desired = "123.5e6"
        self.assertEqual(actual, desired)

        val = -98765e-8
        actual = misc.to_engstr(val)
        desired = "-987.7e-6"
        self.assertEqual(actual, desired)

    def test_to_engstr_zero(self):
        eps = 1e-8
        vals = (0.0, 1e-1 * eps, -1e-1 * eps)

        for val in vals:
            self.assertEqual(misc.to_engstr(val, eps=eps), "0")

    def test_to_engstr_smaller_than_kilo(self):
        val = 123.456
        actual = misc.to_engstr(val)
        desired = "123.5"
        self.assertEqual(actual, desired)

    def test_to_engstr_larger_than_milli(self):
        val = 123.456e-3
        actual = misc.to_engstr(val)
        desired = "0.1235"
        self.assertEqual(actual, desired)

    @unittest.skipIf(True, "FIXME")
    def test_to_engstr_digits(self):
        val = 123456

        digits = 2
        actual = misc.to_engstr(val, digits=digits)
        desired = "12e3"
        self.assertEqual(actual, desired)

        digits = 5
        actual = misc.to_engstr(val, digits=digits)
        desired = "123.46e3"
        self.assertEqual(actual, desired)

    @unittest.skipIf(True, "FIXME")
    def test_to_tuplestr(self):
        seq = ()
        actual = misc.to_tuplestr(seq)
        desired = str(seq)
        self.assertEqual(actual, desired)

        seq = (0.0,)
        actual = misc.to_tuplestr(seq)
        desired = str(seq)
        self.assertEqual(actual, desired)

        seq = (0.0, 1.0)
        actual = misc.to_tuplestr(seq)
        desired = str(seq)
        self.assertEqual(actual, desired)

        seq = (0.0, "a")
        actual = misc.to_tuplestr(seq)
        desired = str(seq)
        self.assertEqual(actual, desired)

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

    def test_get_sha256_hash(self):
        actual = misc.get_sha256_hash(self.default_image_file())
        desired = "7538cbb80cb9103606c48b806eae57d56c885c7f90b9b3be70a41160f9cbb683"
        self.assertEqual(actual, desired)

    def test_get_device(self):
        actual = misc.get_device()
        desired = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(actual, desired)

    def test_get_device_identity(self):
        device = torch.device("mkldnn")
        actual = misc.get_device(device)
        desired = device
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


class TestCuda(PysticheTestCase):
    @staticmethod
    def create_large_cuda_tensor(size_in_gb=256):
        return torch.empty((size_in_gb, *[1024] * 3), device="cuda", dtype=torch.uint8)

    @skip_if_cuda_not_available
    def test_use_cuda_out_of_memory_error(self):
        with self.assertRaises(cuda.CudaOutOfMemoryError):
            with cuda.use_cuda_out_of_memory_error():
                self.create_large_cuda_tensor()

    @skip_if_cuda_not_available
    def test_abort_if_cuda_memory_exausts(self):
        @cuda.abort_if_cuda_memory_exausts
        def fn():
            self.create_large_cuda_tensor()

        with self.assertWarns(cuda.CudaOutOfMemoryWarning):
            fn()
