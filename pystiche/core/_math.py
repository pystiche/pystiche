import warnings

import torch
from torch.nn.functional import relu

from pystiche.misc import build_deprecation_message

__all__ = [
    "nonnegsqrt",
    "possqrt",
    "batch_gram_matrix",
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


def batch_gram_matrix(x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    x_flat = torch.flatten(x, 2)
    gram_matrix = torch.bmm(x_flat, x_flat.transpose(1, 2))
    if normalize:
        gram_matrix /= x_flat.size()[-1]
    return gram_matrix
