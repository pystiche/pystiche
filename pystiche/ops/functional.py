import warnings

import torch
from torch.nn.functional import mse_loss, relu

import pystiche
from pystiche.misc import reduce, build_deprecation_message


def mrf_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    def examplewise_cosine_similarity(
        input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        input = torch.flatten(input, 1)
        input = input / (torch.norm(input, dim=1, keepdim=True) + eps)

        target = torch.flatten(target, 1)
        target = target / (torch.norm(target, dim=1, keepdim=True) + eps)

        return torch.clamp(torch.mm(input, target.t()), max=1.0 / eps)

    with torch.no_grad():
        similarity = examplewise_cosine_similarity(input, target, eps=eps)
        idcs = torch.argmax(similarity, dim=1)
        target = torch.index_select(target, dim=0, index=idcs)
    return mse_loss(input, target, reduction=reduction)


def patch_matching_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    msg = build_deprecation_message(
        "The function patch_matching_loss", "0.4.0", info="It was renamed to mrf_loss"
    )
    warnings.warn(msg, UserWarning)
    return mrf_loss(input, target, eps=eps, reduction=reduction)


def value_range_loss(
    input: torch.Tensor, min: float = 0.0, max: float = 1.0, reduction: str = "mean"
) -> torch.Tensor:
    # TODO: remove sinh call; quadratic, i.e. x * (x-1) is enough
    loss = relu(torch.sinh(input - min) * torch.sinh(input - max))
    return reduce(loss, reduction)


def total_variation_loss(
    input: torch.Tensor, exponent: float = 2.0, reduction: str = "mean"
) -> torch.Tensor:
    # this ignores the last row and column of the image
    grad_vert = input[:, :, :-1, :-1] - input[:, :, 1:, :-1]
    grad_horz = input[:, :, :-1, :-1] - input[:, :, :-1, 1:]
    grad = pystiche.nonnegsqrt(grad_vert ** 2.0 + grad_horz ** 2.0)
    loss = grad ** exponent
    return reduce(loss, reduction)
