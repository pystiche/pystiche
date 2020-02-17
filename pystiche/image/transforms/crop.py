from typing import Any, Union, Tuple, Dict
import torch
from .core import Transform
from . import functional as F

__all__ = [
    "CenterCrop",
    "TopLeftCrop",
    "BottomLeftCrop",
    "TopRightCrop",
    "BottomRightCrop",
]


class TopLeftCrop(Transform):
    def __init__(self, size: Union[Tuple[int, int], int]):
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.top_left_crop(image, self.size)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["size"] = self.size
        return dct


class BottomLeftCrop(Transform):
    def __init__(self, size: Union[Tuple[int, int], int]):
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.bottom_left_crop(image, self.size)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["size"] = self.size
        return dct


class TopRightCrop(Transform):
    def __init__(self, size: Union[Tuple[int, int], int]):
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.top_right_crop(image, self.size)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["size"] = self.size
        return dct


class BottomRightCrop(Transform):
    def __init__(self, size: Union[Tuple[int, int], int]):
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.bottom_right_crop(image, self.size)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["size"] = self.size
        return dct


class CenterCrop(Transform):
    def __init__(self, size: Union[Tuple[int, int], int]):
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.center_crop(image, self.size)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["size"] = self.size
        return dct
