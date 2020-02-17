from typing import Any, Union, Tuple, Dict
import torch
from pystiche.misc import verify_str_arg
from .core import Transform
from . import functional as F

__all__ = [
    "Crop",
    "CenterCrop",
    "TopLeftCrop",
    "BottomLeftCrop",
    "TopRightCrop",
    "BottomRightCrop",
]


class Crop(Transform):
    def __init__(
        self,
        origin: Tuple[int, int],
        size: Union[Tuple[int, int], int],
        vert_anchor: str = "top",
        horz_anchor: str = "left",
    ):
        super().__init__()
        self.origin = origin
        self.size = size
        self.vert_anchor = verify_str_arg(vert_anchor, "vert_anchor", ("top", "bottom"))
        self.horz_anchor = verify_str_arg(horz_anchor, "horz_anchor", ("left", "right"))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.crop(image, self.origin, self.size)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["origin"] = self.origin
        dct["size"] = self.size
        dct["vert_anchor"] = self.vert_anchor
        dct["horz_anchor"] = self.horz_anchor
        return dct


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
        return F.top_left_crop(image, self.size)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["size"] = self.size
        return dct


class TopRightCrop(Transform):
    def __init__(self, size: Union[Tuple[int, int], int]):
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.top_left_crop(image, self.size)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["size"] = self.size
        return dct


class BottomRightCrop(Transform):
    def __init__(self, size: Union[Tuple[int, int], int]):
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.top_left_crop(image, self.size)

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
