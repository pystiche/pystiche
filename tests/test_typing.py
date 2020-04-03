import re
from torch import nn
from pystiche import typing
from utils import PysticheTestCase


def extract_module_names(pattern):
    names = [name for name in nn.__dict__.keys() if name[0].istitle()]
    matching_names = set([name for name in names if pattern.match(name) is not None])
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


class TestTyping(PysticheTestCase):
    def test_is_conv_module(self):
        for module in default_conv_modules():
            msg = (
                f"{module.__class__.__name__} is a conv module, but it is not "
                f"recognized as one."
            )
            self.assertTrue(typing.is_conv_module(module), msg)

        for module in default_pool_modules():
            msg = (
                f"{module.__class__.__name__} is not a conv module, but it is "
                f"recognized as one."
            )
            self.assertFalse(typing.is_conv_module(module), msg)

    def test_is_pool_module(self):
        for module in default_pool_modules():
            msg = (
                f"{module.__class__.__name__} is a pool module, but it is not "
                f"recognized as one."
            )
            self.assertTrue(typing.is_pool_module(module), msg)

        for module in default_conv_modules():
            msg = (
                f"{module.__class__.__name__} is not a pool module, but it is "
                f"recognized as one."
            )
            self.assertFalse(typing.is_pool_module(module), msg)
