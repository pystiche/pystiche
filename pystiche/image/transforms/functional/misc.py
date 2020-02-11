from typing import Optional
import torch
from pystiche.image.utils import force_batched_image
from pystiche.typing import Numeric

__all__ = [
    "normalize",
    "denormalize",
    "float_to_uint8_range",
    "uint8_to_float_range",
    "reverse_channel_order",
    "transform_channels_linearly",
    "transform_channels_affinely",
]


def float_to_uint8_range(x: torch.Tensor) -> torch.Tensor:
    return x * 255.0


def uint8_to_float_range(x: torch.Tensor) -> torch.Tensor:
    return x / 255.0


@force_batched_image
def reverse_channel_order(x: torch.Tensor) -> torch.Tensor:
    return x.flip((1,))


@force_batched_image
def normalize(x: torch.Tensor, mean: Numeric, std: Numeric) -> torch.Tensor:
    return (x - mean) / std


@force_batched_image
def denormalize(x: torch.Tensor, mean: Numeric, std: Numeric) -> torch.Tensor:
    return x * std + mean


@force_batched_image
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


def transform_channels_linearly(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return transform_channels_affinely(x, matrix, bias=None)
