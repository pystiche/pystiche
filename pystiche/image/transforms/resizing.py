from typing import Any, Union, Optional, Tuple, Dict
import torch
from pystiche.image.utils import (
    is_image_size,
    is_edge_size,
)
from .core import Transform
from . import functional as F

__all__ = ["Resize", "Rescale"]


class Resize(Transform):
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        edge: str = "short",
        aspect_ratio: Optional[float] = None,
        interpolation_mode: str = "bilinear",
    ):
        super().__init__()
        self.size = size
        self.edge = edge
        self.aspect_ratio = aspect_ratio
        self.interpolation_mode = interpolation_mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.resize(
            image,
            self.size,
            edge=self.edge,
            aspect_ratio=self.aspect_ratio,
            interpolation_mode=self.interpolation_mode,
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        if is_image_size(self.size):
            dct["image_size"] = self.size
        else:
            key = "edge_size" if is_edge_size(self.size) else "size"
            dct[key] = self.size
            dct["edge"] = self.edge
            if self.aspect_ratio is not None:
                dct["aspect_ratio"] = self.aspect_ratio
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        return dct


class Rescale(Transform):
    def __init__(
        self,
        factor: Union[float, Tuple[float, float]],
        interpolation_mode: str = "bilinear",
    ):
        super().__init__()
        self.factor = factor
        self.interpolation_mode = interpolation_mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.rescale(image, self.factor, self.interpolation_mode)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["factor"] = self.factor
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        return dct
