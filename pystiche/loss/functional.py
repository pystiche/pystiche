from typing import Optional

import torch
from torch.nn.functional import mse_loss, relu

import pystiche
from pystiche.misc import reduce


def mrf_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
    batched_input: Optional[bool] = None,
) -> torch.Tensor:
    r"""Calculates the MRF loss. See :class:`pystiche.ops.MRFOperator` for details.

    Args:
        input: Input of shape :math:`S_1 \times N_1 \times \dots \times N_D`.
        target: Target of shape :math:`S_2 \times N_1 \times \dots \times N_D`.
        eps: Small value to avoid zero division. Defaults to ``1e-8``.
        reduction: Reduction method of the output passed to
            :func:`pystiche.misc.reduce`. Defaults to ``"mean"``.
        batched_input: If ``True``, treat the first dimension of the inputs as batch
            dimension, i.e. :math:`B \times S \times N_1 \times \dots \times N_D`.
            Defaults to ``False``. See :func:`pystiche.cosine_similarity` for details.

    Note:
        The default value of ``batched_input`` will change from ``False`` to ``True``
        in the future.

    Examples:

        >>> input = torch.rand(256, 64, 3, 3)
        >>> target = torch.rand(256, 64, 3, 3)
        >>> score = F.mrf_loss(input, target)

    """
    with torch.no_grad():
        similarity = pystiche.cosine_similarity(
            input, target, eps=eps, batched_input=batched_input
        )
        idcs = torch.argmax(similarity, dim=1)
        target = torch.index_select(target, dim=0, index=idcs)
    return mse_loss(input, target, reduction=reduction)


def value_range_loss(
    input: torch.Tensor, min: float = 0.0, max: float = 1.0, reduction: str = "mean"
) -> torch.Tensor:
    # TODO: remove sinh call; quadratic, i.e. x * (x-1) is enough
    loss = relu(torch.sinh(input - min) * torch.sinh(input - max))
    return reduce(loss, reduction)


def total_variation_loss(
    input: torch.Tensor, exponent: float = 2.0, reduction: str = "mean"
) -> torch.Tensor:
    r"""Calculates the total variation loss. See
    :class:`pystiche.ops.TotalVariationOperator` for details.

    Args:
        input: Input image
        exponent: Parameter :math:`\beta` . A higher value leads to more smoothed
            results. Defaults to ``2.0``.
        reduction: Reduction method of the output passed to
            :func:`pystiche.misc.reduce`. Defaults to ``"mean"``.

    Examples:

        >>> input = torch.rand(2, 3, 256, 256)
        >>> score = F.total_variation_loss(input)
    """
    # this ignores the last row and column of the image
    grad_vert = input[:, :, :-1, :-1] - input[:, :, 1:, :-1]
    grad_horz = input[:, :, :-1, :-1] - input[:, :, :-1, 1:]
    grad = pystiche.nonnegsqrt(grad_vert ** 2.0 + grad_horz ** 2.0)
    loss = grad ** exponent
    return reduce(loss, reduction)
