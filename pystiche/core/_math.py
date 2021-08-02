import warnings
from typing import Optional, cast

import torch
from torch.nn.functional import relu

__all__ = [
    "nonnegsqrt",
    "gram_matrix",
    "cosine_similarity",
]


def nonnegsqrt(x: torch.Tensor) -> torch.Tensor:
    r"""Safely calculates the square-root of a non-negative input

    .. math::

        \fun{nonnegsqrt}{x} =
        \begin{cases}
            \sqrt{x} &\quad\text{if } x \ge 0 \\
            0 &\quad\text{otherwise}
        \end{cases}

    .. note::

        This operation is useful in situations where the input tensor is strictly
        non-negative from a theoretical standpoint, but might be negative due to
        numerical instabilities.

    Args:
        x: Input tensor.
    """
    return torch.sqrt(relu(x))


def gram_matrix(x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    r"""Calculates the channel-wise `Gram matrix
        <https://en.wikipedia.org/wiki/Gramian_matrix>`_ of a batched input tensor.

        Given a tensor :math:`x` of shape
        :math:`B \times C \times N_1 \times \dots \times N_D` each element of the
        single-sample Gram matrix :math:`G_{b,c_1 c_2}` with :math:`b \in 1,\dots,B` and
        :math:`c_1,\,c_2  \in 1,\dots,C` is calculated by

        .. math::

            G_{b,c_1 c_2} = \dotproduct{\fun{vec}{x_{b, c_1}}}{\fun{vec}{x_{b, c_2}}}

        where :math:`\dotproduct{\cdot}{\cdot}` denotes the
        `dot product <https://en.wikipedia.org/wiki/Dot_product>`_ and
        :math:`\fun{vec}{\cdot}` denotes the
        `vectorization function <https://en.wikipedia.org/wiki/Vectorization_(mathematics)>`_ .

        Args:
            x: Input tensor of shape :math:`B \times C \times N_1 \times \dots \times N_D`
            normalize: If True, normalizes the Gram matrix :math:`G` by
                :math:`\prod\limits_{d=1}^{D} N_d` to keep the value range similar for
                different sized inputs. Defaults to ``False``.

        Returns:
            Channel-wise Gram matrix G of shape :math:`B \times C \times C`.
        """
    x_flat = torch.flatten(x, 2)
    gram_matrix = torch.bmm(x_flat, x_flat.transpose(1, 2))
    if not normalize:
        return gram_matrix

    numel = x_flat.size()[-1]
    return gram_matrix / numel


def _norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return cast(torch.Tensor, x / (torch.norm(x, dim=dim, keepdim=True) + eps))


def cosine_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    eps: float = 1e-8,
    batched_input: Optional[bool] = None,
) -> torch.Tensor:
    r"""Calculates the cosine similarity between the samples of ``x1`` and ``x2``.

    Args:
        x1: First input of shape
            :math:`B \times S_1 \times N_1 \times \dots \times N_D`.
        x2: Second input of shape
            :math:`B \times S_2 \times N_1 \times \dots \times N_D`.
        eps: Small value to avoid zero division. Defaults to ``1e-8``.
        batched_input: If ``False``, treat the first dimension of the inputs as sample
            dimension, i.e. :math:`S \times N_1 \times \dots \times N_D`. Defaults to
            ``True``.

    Returns:
        Similarity matrix of shape :math:`B \times S_1 \times S_2` in which every
        element represents the cosine similarity between the corresponding samples
        :math:`S` of ``x1`` and ``x2``. If ``batched_input is False``, the output shape
        is :math:`S_1 \times S_2`

    """
    if batched_input is None:
        msg = (
            "The default value of batched_input has changed "
            "from False to True in version 1.0.0. "
            "To suppress this warning, pass the wanted behavior explicitly."
        )
        warnings.warn(msg, UserWarning)
        batched_input = True

    mm, dim = (torch.bmm, 2) if batched_input else (torch.mm, 1)

    x1 = _norm(torch.flatten(x1, dim), dim=dim, eps=eps)
    x2 = _norm(torch.flatten(x2, dim), dim=dim, eps=eps)
    return torch.clamp(mm(x1, x2.transpose(-1, -2)), max=1.0 / eps)
