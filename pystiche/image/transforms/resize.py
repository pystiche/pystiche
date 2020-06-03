from typing import Any, Dict, Optional, Tuple, Union, cast

import torch

from pystiche.image.utils import is_edge_size, is_image_size

from . import functional as F
from .core import Transform

__all__ = ["Resize", "Rescale"]


class Resize(Transform):
    r"""Resize an image as specified

    Args:
        size: Size of the output. If ``int``, ``edge`` is used to determine the image
            size.
        edge: Corresponding edge if ``size`` is an edge size. Can be ``"short"``,
            ``"long"``, ``"vert"``, or ``"horz"``. Defaults to ``"short"``.
        aspect_ratio: Optional aspect ratio. If ``None`` the aspect ratio of ``image``
            is used. Defaults to ``None``.
        interpolation_mode: Interpolation mode used to resize ``image``. Can be
            ``"nearest"``, ``"bilinear"``, ``"bicubic"``, or ``"area"``. Defaults to
            ``"bilinear"``.
    """

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
        return cast(
            torch.Tensor,
            F.resize(
                image,
                self.size,
                edge=self.edge,
                aspect_ratio=self.aspect_ratio,
                interpolation_mode=self.interpolation_mode,
            ),
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
