import re

import torch
from torch import nn

from pystiche import meta

from .utils import PysticheTestCase


def extract_module_names(pattern):
    names = [name for name in nn.__dict__.keys() if name[0].istitle()]
    matching_names = {name for name in names if pattern.match(name) is not None}
    return matching_names, set(names) - matching_names


def create_module_from_name(name, *args, **kwargs):
    return getattr(nn, name)(*args, **kwargs)


def create_default_conv_module(name, in_channels=1, out_channels=1, kernel_size=1):
    return create_module_from_name(name, in_channels, out_channels, kernel_size)


def default_conv_modules():
    pattern = re.compile("Conv(Transpose)?[1-3]d")
    conv_module_names, _ = extract_module_names(pattern)
    return [create_default_conv_module(name) for name in conv_module_names]


def create_default_pool_module(name, kernel_size=1):
    return create_module_from_name(name, kernel_size)


def default_pool_modules():
    pattern = re.compile("(Adaptive)?(Avg|Max)Pool[1-3]d")
    pool_module_names, _ = extract_module_names(pattern)
    return [create_default_pool_module(name) for name in pool_module_names]


class TestMeta(PysticheTestCase):
    def test_tensor_meta(self):
        tensor_meta = {"dtype": torch.bool, "device": torch.device("cpu")}

        x = torch.empty((), **tensor_meta)

        actual = meta.tensor_meta(x)
        desired = tensor_meta
        self.assertDictEqual(actual, desired)

    def test_tensor_meta_kwargs(self):
        dtype = torch.bool

        x = torch.empty(())

        actual = meta.tensor_meta(x, dtype=dtype)["dtype"]
        desired = dtype
        self.assertEqual(actual, desired)

    def test_is_scalar_tensor(self):
        for scalar_tensor in (torch.tensor(0.0), torch.empty(())):
            self.assertTrue(meta.is_scalar_tensor(scalar_tensor))

        for nd_tensor in (torch.empty(0), torch.empty((0,))):
            self.assertFalse(meta.is_scalar_tensor(nd_tensor))

    def test_is_conv_module(self):
        for module in default_conv_modules():
            msg = (
                f"{module.__class__.__name__} is a conv module, but it is not "
                f"recognized as one."
            )
            self.assertTrue(meta.is_conv_module(module), msg)

        for module in default_pool_modules():
            msg = (
                f"{module.__class__.__name__} is not a conv module, but it is "
                f"recognized as one."
            )
            self.assertFalse(meta.is_conv_module(module), msg)

    def test_conv_module_meta(self):
        conv_module_meta = {
            "kernel_size": (2,),
            "stride": (3,),
            "padding": (4,),
            "dilation": (5,),
        }

        x = nn.Conv1d(1, 1, **conv_module_meta)

        actual = meta.conv_module_meta(x)
        desired = conv_module_meta
        self.assertDictEqual(actual, desired)

    def test_conv_module_meta_kwargs(self):
        stride = (2,)

        x = nn.Conv1d(1, 1, 1, stride=stride)

        actual = meta.conv_module_meta(x, stride=stride)["stride"]
        desired = stride
        self.assertEqual(actual, desired)

    def test_is_pool_module(self):
        for module in default_pool_modules():
            msg = (
                f"{module.__class__.__name__} is a pool module, but it is not "
                f"recognized as one."
            )
            self.assertTrue(meta.is_pool_module(module), msg)

        for module in default_conv_modules():
            msg = (
                f"{module.__class__.__name__} is not a pool module, but it is "
                f"recognized as one."
            )
            self.assertFalse(meta.is_pool_module(module), msg)

    def test_pool_module_meta(self):
        pool_module_meta = {
            "kernel_size": (2,),
            "stride": (3,),
            "padding": (4,),
        }

        x = nn.MaxPool1d(**pool_module_meta)

        actual = meta.pool_module_meta(x)
        desired = pool_module_meta
        self.assertDictEqual(actual, desired)

    def test_pool_module_meta_kwargs(self):
        kernel_size = (2,)

        x = nn.MaxPool1d(kernel_size=kernel_size)

        actual = meta.conv_module_meta(x, kernel_size=kernel_size)["kernel_size"]
        desired = kernel_size
        self.assertEqual(actual, desired)
