import warnings
from typing import Any, Dict, Mapping, Sequence, Tuple, Type

import torch
from torch import nn

import pystiche
from pystiche.image.utils import extract_num_channels
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


class _Normalization(pystiche.Module):
    def __init__(self, mean: Sequence[float], std: Sequence[float],) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    @staticmethod
    def _channel_stats_to_tensor(
        image: torch.Tensor, mean: Sequence[float], std: Sequence[float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_channels = extract_num_channels(image)

        def to_tensor(seq: Sequence[float]) -> torch.Tensor:
            if len(seq) != num_channels:
                msg = (
                    f"The length of the channel statistics and the number of image "
                    f"channels do not match: {len(seq)} != {num_channels}"
                )
                raise RuntimeError(msg)
            return torch.tensor(seq, device=image.device).view(1, -1, 1, 1)

        return to_tensor(mean), to_tensor(std)

    @staticmethod
    def _format_stats(stats: Sequence[float], fmt: str = "{:g}") -> str:
        return str(tuple(fmt.format(stat) for stat in stats))

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["mean"] = self._format_stats(self.mean)
        dct["std"] = self._format_stats(self.std)
        return dct


class Normalize(_Normalization):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mean, std = self._channel_stats_to_tensor(image, self.mean, self.std)
        return image.sub(mean).div(std)


class Denormalize(_Normalization):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mean, std = self._channel_stats_to_tensor(image, self.mean, self.std)
        return image.mul(std).add(mean)


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
