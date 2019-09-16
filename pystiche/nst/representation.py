from typing import Union, Sequence
import torch
import pystiche


def calculate_gram_repr(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    return pystiche.channelwise_gram_matrix(x, normalize=normalize)


def calculate_mrf_repr(
    x: torch.Tensor,
    patch_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int] = 1,
) -> torch.Tensor:
    return pystiche.extract_patches2d(x, patch_size, stride)
