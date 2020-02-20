import torch
import pystiche
from pystiche.image.utils import force_batched_image
from ._misc import transform_channels_linearly

__all__ = [
    "rgb_to_grayscale",
    "grayscale_to_fakegrayscale",
    "rgb_to_fakegrayscale",
    "grayscale_to_binary",
    "rgb_to_binary",
    "rgb_to_yuv",
    "yuv_to_rgb",
]


def rgb_to_grayscale(x: torch.Tensor) -> torch.Tensor:
    transformation_matrix = torch.tensor(
        ((0.299, 0.587, 0.114),), **pystiche.tensor_meta(x)
    )
    return transform_channels_linearly(x, transformation_matrix)


@force_batched_image
def grayscale_to_fakegrayscale(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(1, 3, 1, 1)


def rgb_to_fakegrayscale(x: torch.Tensor) -> torch.Tensor:
    return grayscale_to_fakegrayscale(rgb_to_grayscale(x))


def grayscale_to_binary(x: torch.Tensor) -> torch.Tensor:
    # TODO: add static / dynamic thresholding
    return torch.round(x)


def rgb_to_binary(x: torch.Tensor) -> torch.Tensor:
    return grayscale_to_binary(rgb_to_grayscale(x))


def rgb_to_yuv(x: torch.Tensor) -> torch.Tensor:
    transformation_matrix = torch.tensor(
        ((0.299, 0.587, 0.114), (-0.147, -0.289, 0.436), (0.615, -0.515, -0.100)),
        **pystiche.tensor_meta(x)
    )
    return transform_channels_linearly(x, transformation_matrix)


def yuv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    transformation_matrix = torch.tensor(
        ((1.000, 0.000, 1.140), (1.000, -0.395, -0.581), (1.000, 2.032, 0.000)),
        **pystiche.tensor_meta(x)
    )
    return transform_channels_linearly(x, transformation_matrix)
