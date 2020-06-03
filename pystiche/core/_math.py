import warnings
from typing import cast

import torch
from torch.nn.functional import relu

from pystiche.misc import build_deprecation_message

__all__ = [
    "nonnegsqrt",
    "possqrt",
    "gram_matrix",
    "batch_gram_matrix",
    "cosine_similarity",
]


def nonnegsqrt(x: torch.Tensor) -> torch.Tensor:
    r"""Safely calculates the square-root of a non-negative input

    .. math::

        \text{nonnegsqrt}\left( x \right) =
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


def possqrt(x: torch.Tensor) -> torch.Tensor:
    msg = build_deprecation_message(
        "The function possqrt", "0.4.0", info="It was renamed to nonnegsqrt"
    )
    warnings.warn(msg, UserWarning)
    return nonnegsqrt(x)


def gram_matrix(x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    r"""Calculates the channel-wise `Gram matrix
        <https://en.wikipedia.org/wiki/Gramian_matrix>`_ of a batched input tensor.

        Given a tensor :math:`x` of shape
        :math:`B \times C \times N_1 \times \dots \times N_D` each element of the
        single-sample Gram matrix :math:`G_{b,c_1 c_2}` with :math:`b \in 1,\dots,B` and
        :math:`c1,\,c2  \in 1,\dots,C` is calculated by

        .. math::

            G_{b,c_1 c_2} = \left\langle \text{vec}\left(x_{b, c_1}\right) ,\, \text{vec}\left(x_{b, c_2}\right) \right\rangle

        where :math:`\left\langle \cdot ,\, \cdot \right\rangle` denotes the
        `dot product <https://en.wikipedia.org/wiki/Dot_product>`_ and
        :math`\text{vec}\left(\cdot\right)` denotes the
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


def batch_gram_matrix(x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    msg = build_deprecation_message(
        "The function batch_gram_matrix", "0.4.0", info="It was renamed to gram_matrix"
    )
    warnings.warn(msg, UserWarning)
    return gram_matrix(x, normalize=normalize)


def _norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return cast(torch.Tensor, x / (torch.norm(x, dim=1, keepdim=True) + eps))


def cosine_similarity(
    input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    r"""Calculates the cosine similarity between the samples of ``x1`` and ``x2``.

    Args:
        input: First input of shape :math:`B_1 \times N_1 \times \dots \times N_D`.
        target: Second input of shape :math:`B_2 \times N_1 \times \dots \times N_D`.
        eps: Small value to avoid zero division. Defaults to ``1e-8``.

    Returns:
        Similarity matrix of shape :math:`B_1 \times B_2` in which every element
        represents the cosine similarity between the corresponding samples of ``x1``
        and ``x2``.
    """
    input = _norm(torch.flatten(input, 1), eps=eps)
    target = _norm(torch.flatten(target, 1), eps=eps)
    return torch.clamp(torch.mm(input, target.t()), max=1.0 / eps)
