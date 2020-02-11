from typing import Optional, Tuple
from abc import abstractmethod
from copy import copy
import torch
from pystiche.typing import Numeric
from pystiche.misc import to_engstr
from pystiche.image import edge_to_image_size, extract_image_size, extract_aspect_ratio
from .. import functional as F
from .core import Transform

__all__ = ["ResizeTransform", "Resize", "FixedAspectRatioResize", "Rescale"]


class ResizeTransform(Transform):
    def __init__(self, interpolation_mode: str = "bilinear") -> None:
        super().__init__()
        self.interpolation_mode: str = interpolation_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_size = self.calculate_image_size(x)
        return F.resize(x, image_size, interpolation_mode=self.interpolation_mode)

    @abstractmethod
    def calculate_image_size(self, x: torch.Tensor) -> Tuple[int, int]:
        pass

    @property
    @abstractmethod
    def has_fixed_size(self) -> bool:
        pass

    def extra_repr(self) -> str:
        extras = []
        resize_transform_extras = self.extra_resize_transform_repr()
        if resize_transform_extras:
            extras.append(resize_transform_extras)
        if self.interpolation_mode != "bilinear":
            extras.append(", interpolation_mode={interpolation_mode}")
        return ", ".join(extras).format(**self.__dict__)

    def extra_resize_transform_repr(self) -> str:
        return ""


class Resize(ResizeTransform):
    def __init__(self, image_size: Tuple[int, int], **kwargs):
        super().__init__(**kwargs)
        self.image_size: Tuple[int, int] = image_size

    def calculate_image_size(self, x: torch.Tensor) -> Tuple[int, int]:
        return self.image_size

    @property
    def has_fixed_size(self) -> bool:
        return True

    def extra_resize_transform_repr(self):
        return "image_size={image_size}".format(**self.__dict__)


class FixedAspectRatioResize(ResizeTransform):
    def __init__(
        self,
        edge_size: int,
        edge: str = "short",
        aspect_ratio: Optional[Numeric] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edge_size: int = edge_size
        self.edge: str = edge

        self.aspect_ratio: Optional[Numeric] = aspect_ratio
        if aspect_ratio is not None:
            self.image_size = edge_to_image_size(edge_size, aspect_ratio, edge)
        else:
            self.image_size = None

    def calculate_image_size(self, x: torch.Tensor) -> Tuple[int, int]:
        if self.has_fixed_size:
            return self.image_size
        else:
            aspect_ratio = extract_aspect_ratio(x)
            return edge_to_image_size(self.edge_size, aspect_ratio, self.edge)

    @property
    def has_fixed_size(self) -> bool:
        return self.image_size is not None

    def extra_resize_transform_repr(self) -> str:
        if self.has_fixed_size:
            return "size={size}".format(**self.__dict__)
        else:
            dct = copy(self.__dict__)
            dct["aspect_ratio"] = to_engstr(dct["aspect_ratio"])
            extras = (
                "edge_size={edge_size}",
                "aspect_ratio={aspect_ratio}",
                "edge={edge}",
            )
            return ", ".join(extras).format(**dct)


class Rescale(ResizeTransform):
    def __init__(self, factor: Numeric, **kwargs):
        super().__init__(**kwargs)
        self.factor: Numeric = factor

    def calculate_image_size(self, x):
        return [round(edge_size * self.factor) for edge_size in extract_image_size(x)]

    @property
    def has_fixed_size(self) -> bool:
        return False

    def extra_resize_transform_repr(self) -> str:
        return "factor={factor}".format(**self.__dict__)
