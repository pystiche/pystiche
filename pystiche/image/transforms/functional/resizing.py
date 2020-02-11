from typing import Sequence
import torch
from torch.nn.functional import interpolate
from pystiche.image.utils import force_batched_image
from ._utils import get_align_corners


@force_batched_image
def resize(
    x: torch.Tensor, size: Sequence[int], interpolation_mode: str = "bilinear"
) -> torch.Tensor:
    return interpolate(
        x,
        size=size,
        scale_factor=None,
        mode=interpolation_mode,
        align_corners=get_align_corners(interpolation_mode),
    )
