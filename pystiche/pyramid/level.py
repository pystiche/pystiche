from typing import Optional
import torch
from pystiche import Object
from pystiche.misc import verify_str_arg
from pystiche.image.transforms.functional import resize

__all__ = ["PyramidLevel"]


class PyramidLevel(Object):
    def __init__(self, edge_size: int, num_steps: int, edge: str):
        self.edge_size = edge_size
        self.num_steps: int = num_steps
        self.edge = verify_str_arg(edge, "edge", ("short", "long"))

    def _resize(
        self,
        image: torch.Tensor,
        aspect_ratio: Optional[float],
        interpolation_mode: str,
    ) -> torch.Tensor:

        with torch.no_grad():
            image = resize(
                image,
                self.edge_size,
                edge=self.edge,
                aspect_ratio=aspect_ratio,
                interpolation_mode=interpolation_mode,
            )
        return image.detach()

    def resize_image(
        self,
        image: torch.Tensor,
        aspect_ratio: Optional[float] = None,
        interpolation_mode: str = "bilinear",
    ) -> torch.Tensor:
        return self._resize(image, aspect_ratio, interpolation_mode)

    def resize_guide(
        self,
        guide: torch.Tensor,
        aspect_ratio: Optional[float] = None,
        interpolation_mode: str = "nearest",
    ) -> torch.Tensor:
        return self._resize(guide, aspect_ratio, interpolation_mode)

    def __iter__(self):
        for step in range(1, self.num_steps + 1):
            yield step

    def _properties(self):
        dct = super()._properties()
        dct["edge_size"] = self.edge_size
        dct["num_steps"] = self.num_steps
        dct["edge"] = self.edge
        return dct
