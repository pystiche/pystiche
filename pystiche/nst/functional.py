from typing import Union, Sequence
import torch
from torch.nn.functional import mse_loss, relu
import pystiche
from pystiche.typing import Numeric
from pystiche.misc import verify_str_arg
from .representation import calculate_gram_repr, calculate_mrf_repr


def _reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    verify_str_arg(reduction, "reduction", ("mean", "sum", "none"))
    if reduction == "mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:  # reduction == "none":
        return x


def patch_matching_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    with torch.no_grad():
        similarity = pystiche.examplewise_cosine_similarity(input, target, eps=eps)
        idcs = torch.argmax(similarity, dim=1)
        target = torch.index_select(target, dim=0, index=idcs)
    return mse_loss(input, target, reduction=reduction)


def value_range_loss(
    input: torch.Tensor, min=0.0, max=1.0, reduction: str = "mean"
) -> torch.Tensor:
    loss = relu(torch.sinh(input - min) * torch.sinh(input - max))
    return _reduce(loss, reduction)


def total_variation_loss(
    input: torch.Tensor, exponent: Numeric = 2.0, reduction: str = "mean"
) -> torch.Tensor:
    # this ignores the last row and column of the image
    grad_vert = input[:, :, :-1, :-1] - input[:, :, 1:, :-1]
    grad_horz = input[:, :, :-1, :-1] - input[:, :, :-1, 1:]
    grad = pystiche.safesqrt(grad_vert ** 2.0 + grad_horz ** 2.0)
    return _reduce(grad ** exponent, reduction)


def direct_encoding_loss(
    input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    return mse_loss(input, target, reduction=reduction)


def gram_loss(
    input: torch.Tensor, target, normalize: bool, reduction: str = "mean"
) -> torch.Tensor:
    input = calculate_gram_repr(input, normalize=normalize)
    target = calculate_gram_repr(target, normalize=normalize)
    return mse_loss(input, target, reduction=reduction)


def mrf_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    patch_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int] = 1,
    reduction: str = "mean",
) -> torch.Tensor:
    input = calculate_mrf_repr(input, patch_size, stride=stride)
    target = calculate_mrf_repr(target, patch_size, stride=stride)
    return patch_matching_loss(input, target, reduction=reduction)
