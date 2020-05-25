from typing import cast

import torch

from . import functional as F
from .core import Transform

__all__ = [
    "RGBToGrayscale",
    "GrayscaleToFakegrayscale",
    "RGBToFakegrayscale",
    "GrayscaleToBinary",
    "RGBToBinary",
    "RGBToYUV",
    "YUVToRGB",
]


class RGBToGrayscale(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rgb_to_grayscale(x)


class GrayscaleToFakegrayscale(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, F.grayscale_to_fakegrayscale(x))


class RGBToFakegrayscale(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rgb_to_fakegrayscale(x)


class GrayscaleToBinary(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.grayscale_to_binary(x)


class RGBToBinary(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rgb_to_binary(x)


class RGBToYUV(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rgb_to_yuv(x)


class YUVToRGB(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.yuv_to_rgb(x)
