from typing import Optional
import torch
from pystiche.misc import verify_str_arg, build_obj_str
from pystiche.image.transforms import FixedAspectRatioResize

__all__ = ["PyramidLevel"]


class PyramidLevel:
    def __init__(self, edge_size: int, num_steps: int, edge: str):

        self.num_steps: int = num_steps
        self.edge_size = edge_size
        self.edge = verify_str_arg(edge, "edge", ("short", "long"))

    def _resize(
        self,
        image: torch.Tensor,
        aspect_ratio: Optional[float],
        interpolation_mode: str,
    ) -> torch.Tensor:
        transform = FixedAspectRatioResize(
            edge_size=self.edge_size,
            edge=self.edge,
            aspect_ratio=aspect_ratio,
            interpolation_mode=interpolation_mode,
        )
        with torch.no_grad():
            resized_image = transform(image)
        return resized_image.detach()

    def resize_image(
        self,
        image: torch.Tensor,
        aspect_ratio: Optional[float] = None,
        interpolation_mode: str = "bilinear",
    ) -> torch.Tensor:
        return self._resize(image, aspect_ratio, interpolation_mode)

    def resize_guide(
        self, guide: torch.Tensor, aspect_ratio: Optional[float] = None
    ) -> torch.Tensor:
        return self._resize(guide, aspect_ratio, interpolation_mode="nearest")

    def __iter__(self):
        for step in range(self.num_steps):
            yield step

    def __str__(self):
        name = self.__class__.__name__
        description = ", ".join(
            [
                f"num_steps={self.num_steps}",
                f"edge_size={self.edge_size}",
                f"edge={self.edge}",
            ]
        )
        return build_obj_str(name, description=description)
