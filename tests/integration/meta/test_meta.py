import re

import pytest

import torch
from torch import nn

from pystiche import meta


class TestTensorMeta:
    def test_main(self):
        tensor_meta = {"dtype": torch.bool, "device": torch.device("cpu")}

        x = torch.empty((), **tensor_meta)

        actual = meta.tensor_meta(x)
        desired = tensor_meta
        assert actual == desired


    def test_kwargs(main):
        dtype = torch.bool

        x = torch.empty(())

        actual = meta.tensor_meta(x, dtype=dtype)["dtype"]
        desired = dtype
        assert actual == desired


def test_is_scalar_tensor():
    for scalar_tensor in (torch.tensor(0.0), torch.empty(())):
        assert meta.is_scalar_tensor(scalar_tensor)

    for nd_tensor in (torch.empty(0), torch.empty((0,))):
        assert not meta.is_scalar_tensor(nd_tensor)


def make_nn_module(name_or_cls, *args, **kwargs):
    cls = getattr(nn, name_or_cls) if isinstance(name_or_cls, str) else name_or_cls
    return cls(*args, **kwargs)


def extract_nn_module_names(pattern):
    names = [name for name in nn.__dict__.keys() if name[0].istitle()]
    matching_names = {name for name in names if pattern.match(name) is not None}
    return matching_names, set(names) - matching_names


def make_conv_module(
    name_or_cls, in_channels=1, out_channels=1, kernel_size=1, **kwargs
):
    return make_nn_module(name_or_cls, in_channels, out_channels, kernel_size, **kwargs)


@pytest.fixture
def conv_modules():
    pattern = re.compile("Conv(Transpose)?[1-3]d")
    conv_module_names, _ = extract_nn_module_names(pattern)
    return [make_conv_module(name) for name in conv_module_names]


def make_pool_module(name_or_cls, kernel_size=1, **kwargs):
    return make_nn_module(name_or_cls, kernel_size, **kwargs)


@pytest.fixture
def pool_modules():
    pattern = re.compile("(Adaptive)?(Avg|Max)Pool[1-3]d")
    pool_module_names, _ = extract_nn_module_names(pattern)
    return [make_pool_module(name) for name in pool_module_names]


def test_is_conv_module(conv_modules, pool_modules):
    for module in conv_modules:
        msg = (
            f"{module.__class__.__name__} is a conv module, but it is not "
            f"recognized as one."
        )
        assert meta.is_conv_module(module), msg

    for module in pool_modules:
        msg = (
            f"{module.__class__.__name__} is not a conv module, but it is "
            f"recognized as one."
        )
        assert not meta.is_conv_module(module), msg


class TestConvModuleMeta:
    def test_main(self):
        conv_module_meta = {
            "kernel_size": (2,),
            "stride": (3,),
            "padding": (4,),
            "dilation": (5,),
        }
        x = make_conv_module(nn.Conv1d, **conv_module_meta)

        actual = meta.conv_module_meta(x)
        desired = conv_module_meta
        assert actual == desired


    def test_kwargs(self):
        stride = (2,)
        x = make_conv_module(nn.Conv2d, stride=stride)

        actual = meta.conv_module_meta(x, stride=stride)["stride"]
        desired = stride
        assert actual == desired


def test_is_pool_module(pool_modules, conv_modules):
    for module in pool_modules:
        msg = (
            f"{module.__class__.__name__} is a pool module, but it is not "
            f"recognized as one."
        )
        assert meta.is_pool_module(module), msg

    for module in conv_modules:
        msg = (
            f"{module.__class__.__name__} is not a pool module, but it is "
            f"recognized as one."
        )
        assert not meta.is_pool_module(module), msg


class TestPoolModuleMeta:
    def test_main(self):
        pool_module_meta = {
            "kernel_size": (2,),
            "stride": (3,),
            "padding": (4,),
        }
        x = make_pool_module(nn.MaxPool1d, **pool_module_meta)

        actual = meta.pool_module_meta(x)
        desired = pool_module_meta
        assert actual == desired


    def test_kwargs(self):
        kernel_size = (2,)
        x = make_pool_module(nn.MaxPool1d, kernel_size=kernel_size)

        actual = meta.pool_module_meta(x, kernel_size=kernel_size)["kernel_size"]
        desired = kernel_size
        assert actual == desired
