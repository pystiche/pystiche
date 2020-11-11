import warnings
from typing import Mapping, Type

import torch
from torch import nn

import pystiche
from pystiche.image._transforms import Denormalize, Normalize
from pystiche.misc import build_deprecation_message

__all__ = [
    "TorchPreprocessing",
    "TorchPostprocessing",
    "CaffePreprocessing",
    "CaffePostprocessing",
    "preprocessing",
    "get_preprocessor",
    "postprocessing",
]


TORCH_MEAN = (0.485, 0.456, 0.406)
TORCH_STD = (0.229, 0.224, 0.225)


class TorchPreprocessing(pystiche.SequentialModule):
    def __init__(self) -> None:
        transforms = (Normalize(TORCH_MEAN, TORCH_STD),)
        super().__init__(*transforms)


class TorchPostprocessing(pystiche.SequentialModule):
    def __init__(self) -> None:
        transforms = (Denormalize(TORCH_MEAN, TORCH_STD),)
        super().__init__(*transforms)


CAFFE_MEAN = (0.485, 0.458, 0.408)
CAFFE_STD = (1.0, 1.0, 1.0)


class FloatToUint8Range(pystiche.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.mul(255)


class Uint8ToFloatRange(pystiche.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.div(255)


class ReverseChannelOrder(pystiche.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.flip(1)


class CaffePreprocessing(pystiche.SequentialModule):
    def __init__(self) -> None:
        transforms = (
            Normalize(CAFFE_MEAN, CAFFE_STD),
            FloatToUint8Range(),
            ReverseChannelOrder(),
        )
        super().__init__(*transforms)


class CaffePostprocessing(pystiche.SequentialModule):
    def __init__(self) -> None:
        transforms = (
            ReverseChannelOrder(),
            Uint8ToFloatRange(),
            Denormalize(CAFFE_MEAN, CAFFE_STD),
        )
        super().__init__(*transforms)


_PREPROCESSING = {"torch": TorchPreprocessing, "caffe": CaffePreprocessing}
_POSTPROCESSING = {"torch": TorchPostprocessing, "caffe": CaffePostprocessing}


def _processing(framework: str, dct: Mapping[str, Type[nn.Module]]) -> nn.Module:
    try:
        return dct[framework]()
    except KeyError as error:
        msg = (
            f"""No processing available for framework '{framework}'. """
            f"""Available frameworks are '{"', '".join(sorted(dct.keys()))}'"""
        )
        raise ValueError(msg) from error


def preprocessing(framework: str) -> nn.Module:
    return _processing(framework, _PREPROCESSING)


def get_preprocessor(framework: str) -> nn.Module:
    msg = build_deprecation_message(
        "The function 'get_preprocessor'",
        "1.0",
        info="It was renamed to 'preprocessing'.",
    )
    warnings.warn(msg)
    return preprocessing(framework)


def postprocessing(framework: str) -> nn.Module:
    return _processing(framework, _POSTPROCESSING)
