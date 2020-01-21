from typing import Optional, Sequence
from PIL import Image
import torch
from torch.nn.functional import interpolate
from torchvision.transforms.functional import (
    to_tensor as _to_tensor,
    to_pil_image as _to_pil_image,
)
from pystiche.typing import Numeric

__all__ = [
    "import_from_pil",
    "export_to_pil",
    "normalize",
    "denormalize",
    "float_to_uint8_range",
    "uint8_to_float_range",
    "reverse_channel_order",
    "resize",
    "transform_channels_linearly",
    "transform_channels_affinely",
]


def import_from_pil(image: Image, device: torch.device) -> torch.Tensor:
    return _to_tensor(image).unsqueeze(0).to(device)


def export_to_pil(tensor: torch.Tensor, mode: Optional[str] = None) -> Image:
    return _to_pil_image(tensor.detach().cpu().squeeze(0).clamp(0.0, 1.0), mode)


def float_to_uint8_range(x: torch.Tensor) -> torch.Tensor:
    return x * 255.0


def uint8_to_float_range(x: torch.Tensor) -> torch.Tensor:
    return x / 255.0


def reverse_channel_order(x: torch.Tensor) -> torch.Tensor:
    return x.flip((1,))


def denormalize(x: torch.Tensor, mean: Numeric, std: Numeric) -> torch.Tensor:
    return x * std + mean


def normalize(x: torch.Tensor, mean: Numeric, std: Numeric) -> torch.Tensor:
    return (x - mean) / std


def resize(
    x: torch.Tensor, size: Sequence[int], interpolation_mode: str = "bilinear"
) -> torch.Tensor:
    if interpolation_mode in ("nearest", "area"):
        align_corners = None
    else:
        align_corners = False

    return interpolate(
        x,
        size=size,
        scale_factor=None,
        mode=interpolation_mode,
        align_corners=align_corners,
    )


def transform_channels_linearly(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return transform_channels_affinely(x, matrix, bias=None)


def transform_channels_affinely(
    x: torch.Tensor, matrix: torch.Tensor, bias: Optional[torch.tensor] = None
) -> torch.Tensor:
    batch_size, _, *spatial_size = x.size()
    x = torch.flatten(x, 2)

    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    num_channels = matrix.size()[1]

    x = torch.bmm(matrix, x)

    if bias is not None:
        if bias.dim() == 2:
            bias = bias.unsqueeze(0)
        x += bias

    return x.view(batch_size, num_channels, *spatial_size)
