from typing import Any, Dict, Tuple, Union, cast

import torch

from pystiche.image import extract_image_size

from . import functional as F
from .core import Transform
from .functional._crop import _parse_size

__all__ = [
    "Crop",
    "CenterCrop",
    "TopLeftCrop",
    "BottomLeftCrop",
    "TopRightCrop",
    "BottomRightCrop",
    "ValidRandomCrop",
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
        self.vert_anchor = vert_anchor
        self.horz_anchor = horz_anchor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            F.crop(
                image,
                self.origin,
                self.size,
                vert_anchor=self.vert_anchor,
                horz_anchor=self.horz_anchor,
            ),
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["origin"] = self.origin
        dct["size"] = self.size
        if self.vert_anchor != "top" or self.horz_anchor != "left":
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


class ValidRandomCrop(Transform):
    def __init__(self, size: Union[Tuple[int, int], int]):
        super().__init__()
        self.size = size

    @staticmethod
    def get_random_origin(
        image_size: Tuple[int, int], crop_size: Union[Tuple[int, int], int]
    ) -> Tuple[int, int]:
        image_height, image_width = image_size
        crop_height, crop_width = _parse_size(crop_size)

        def randint(range: int) -> int:
            if range < 0:
                msg = "The crop size has to be smaller or equal to the image size."
                raise RuntimeError(msg)
            return cast(int, torch.randint(range + 1, (), dtype=torch.long).item())

        vert_origin = randint(image_height - crop_height)
        horz_origin = randint(image_width - crop_width)
        return (vert_origin, horz_origin)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        origin = self.get_random_origin(extract_image_size(image), self.size)
        return cast(torch.Tensor, F.crop(image, origin, self.size))

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["size"] = self.size
        return dct
