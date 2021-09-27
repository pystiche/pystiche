import os
import unittest.mock
from distutils import dir_util

import pytorch_testing_utils as ptu

from torch import autograd, nn

import pystiche

__all__ = [
    "make_mock_target",
    "ContextMock",
    "patch_models_load_state_dict_from_url",
    "patch_home",
    "ModuleMock",
]

DEFAULT_MOCKER = unittest.mock


def make_mock_target(*args, pkg="pystiche"):
    return ".".join((pkg, *args))


class ContextMock:
    def __init__(self, mock):
        self.__mock__ = mock

    def __getattribute__(self, item):
        if item == "__mock__":
            return object.__getattribute__(self, "__mock__")
        else:
            return getattr(self.__mock__, item)

    def __setattr__(self, key, value):
        if key == "__mock__":
            object.__setattr__(self, key, value)
        else:
            self.__mock__.__setattr__(key, value)

    def __call__(self, *args, **kwargs):
        return self.__mock__(*args, **kwargs)

    def __enter__(self):
        self.__mock__.reset_mock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def patch_models_load_state_dict_from_url(mocker=DEFAULT_MOCKER):
    return ContextMock(
        mocker.patch(
            make_mock_target(
                "enc",
                "models",
                "utils",
                "ModelMultiLayerEncoder",
                "load_state_dict_from_url",
            )
        )
    )


def patch_home(home, copy=True, mocker=DEFAULT_MOCKER):
    if copy:
        dir_util.copy_tree(pystiche.home(), home)

    return mocker.patch.dict(os.environ, values={"PYSTICHE_HOME": home})


class NoOpFunction(autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        input, *args = args
        return input

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, *grad_outputs = grad_outputs
        grad_input = (grad_output, *[None] * len(grad_outputs))
        return grad_input


class ModuleMock(nn.Module):
    def __init__(self, *methods):
        super().__init__()
        self._call_args_list = []
        for method in methods:
            setattr(
                self,
                method,
                unittest.mock.MagicMock(name=f"{type(self).__name__}.{method}"),
            )

    def forward(self, *args, **kwargs):
        self._call_args_list.insert(0, (args, kwargs))
        return NoOpFunction.apply(*args, **kwargs)

    @property
    def call_args_list(self):
        return self._call_args_list

    @property
    def called(self):
        return bool(self.call_args_list)

    @property
    def call_args(self):
        return self.call_args_list[0]

    @property
    def call_count(self):
        return len(self.call_args_list)

    def assert_called(self, count=None):
        assert self.called
        if count is not None:
            assert self.call_count == count

    def assert_called_with(self, input, *args, **kwargs):
        self.assert_called()
        (input_, *args_), kwargs_ = self.call_args
        ptu.assert_allclose(input_, input)
        assert tuple(args_) == args
        assert kwargs_ == kwargs

    def assert_called_once(self):
        self.assert_called(count=1)

    def assert_called_once_with(self, input, *args, **kwargs):
        self.assert_called_once()
        self.assert_called_with(input, *args, **kwargs)
