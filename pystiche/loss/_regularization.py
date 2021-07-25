from typing import Any, Dict, Optional

import torch

from . import functional as F
from ._loss import RegularizationLoss

__all__ = ["TotalVariationLoss", "ValueRangeLoss"]


class TotalVariationLoss(RegularizationLoss):
    r"""The total variation loss is a regularizer used to suppress checkerboard
    artifacts by penalizing the gradient of the image. It is calculated by

    .. math::

        \mean[\limits_{i,j}] \parentheses{\parentheses{x_{i,j+1} - x_{i,j}}^2 + \parentheses{x_{i+1,j} - x_{i,j}}^2}^{\frac{\beta}{2}}

    where :math:`x` denotes the image and :math:`i,j` index a specific pixel.

    .. note::

        Opposed to the paper, the implementation calculates the grand average
        :math:`\mean` opposed to the grand sum :math:`\sum` to account for different
        sized images.

    Args:
        exponent: Parameter :math:`\beta` . A higher value leads to more smoothed
            results. Defaults to ``2.0``.
        score_weight: Score weight of the operator. Defaults to ``1.0``.

    Examples:

        >>> loss = pystiche.loss.TotalVariationLoss()
        >>> input = torch.rand(2, 3, 256, 256)
        >>> score = loss(input)

    .. seealso::

        The total variation loss was introduced by Mahendran and Vedaldi in
        :cite:`MV2015` .
    """

    def __init__(
        self,
        *,
        exponent: float = 2.0,
        input_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1.0,
    ):
        super().__init__(input_guide=input_guide, score_weight=score_weight)
        self.exponent = exponent

    def input_enc_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.total_variation_loss(input_repr, exponent=self.exponent)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["exponent"] = f"{self.exponent:g}"
        return dct


class ValueRangeLoss(RegularizationLoss):
    def input_enc_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.value_range_loss(input_repr)
