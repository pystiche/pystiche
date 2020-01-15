import torch
import pystiche
from pystiche.misc import verify_str_arg

__all__ = ["PyramidLevel"]


class PyramidLevel(pystiche.Object):
    def __init__(
        self,
        edge_size: int,
        num_steps: int,
        edge: str,
        interpolation_mode: str = "bilinear",
    ):

        self.num_steps: int = num_steps
        self.edge_size = edge_size

        self.edge = verify_str_arg(edge, "edge", ("short", "long"))
        self.interpolation_mode = interpolation_mode

    def resize_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def resize_guide(self, guide: torch.Tensor) -> torch.Tensor:
        pass

    def __iter__(self):
        for step in range(self.num_steps):
            yield step

    def extra_str(self) -> str:
        extras = ["num_steps={num_steps}", "edge_size={edge_size}", "edge={edge}"]
        if self.interpolation_mode != "bilinear":
            extras.append("interpolation_mode={interpolation_mode}")
        return ", ".join(extras).format(**self.__dict__)
