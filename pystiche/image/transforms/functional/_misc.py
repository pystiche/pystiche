from typing import Optional, Sequence, Tuple, cast

import torch

from pystiche.image.utils import extract_num_channels, force_batched_image

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


def _channel_stats_to_tensor(
    image: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_channels = extract_num_channels(image)

    def to_tensor(seq: Sequence[float]) -> torch.Tensor:
        if len(seq) != num_channels:
            msg = (
                f"The length of the channel statistics and the number of image "
                f"channels do not match: {len(seq)} != {num_channels}"
            )
            raise RuntimeError(msg)
        return torch.tensor(seq, device=image.device).view(1, -1, 1, 1)

    return to_tensor(mean), to_tensor(std)


@force_batched_image
def normalize(
    x: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> torch.Tensor:
    mean, std = _channel_stats_to_tensor(x, mean, std)
    return (x - mean) / std


@force_batched_image
def denormalize(
    x: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> torch.Tensor:
    mean, std = _channel_stats_to_tensor(x, mean, std)
    return x * std + mean


@force_batched_image
def transform_channels_affinely(
    x: torch.Tensor, matrix: torch.Tensor, bias: Optional[torch.Tensor] = None
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
    return cast(torch.Tensor, transform_channels_affinely(x, matrix, bias=None))
