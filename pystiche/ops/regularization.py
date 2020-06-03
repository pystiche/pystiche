from typing import Any, Dict

import torch

from pystiche.misc import is_almost, to_engstr

from . import functional as F
from .op import PixelRegularizationOperator

__all__ = ["TotalVariationOperator", "ValueRangeOperator"]


class TotalVariationOperator(PixelRegularizationOperator):
    r"""The total variation loss is a regularizer used to suppress checkerboard
    artifacts by penalizing the gradient of the image. It is calculated by

    .. math::

        \overline{\sum}\limits_{i, j} \left( \left( x_{i,j+1} - x_{ij} \right)^2 + \left( x_{i+1,j} - x_{ij} \right)^2 \right)^{\frac{\beta}{2}}

    where :math:`x` denotes the image and :math:`i,j` index a specific pixel.

    .. note::

        Opposed to the paper, the implementation calculates the grand average
        (:math:`\overline{\sum}`) opposed to the grand sum (:math:`\sum`) to account
        for different sized images.

    Args:
        exponent: Parameter :math:`\beta` . A higher value leads to more smoothed
            results. Defaults to ``2.0``.
        score_weight: Score weight of the operator. Defaults to ``1.0``.

    .. seealso::

        The total variation loss was introduced by Mahendran and Vedaldi in
        :cite:`MV2014` .
    """

    def __init__(self, exponent: float = 2.0, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight)
        self.exponent = exponent

    def input_image_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.total_variation_loss(input_repr, exponent=self.exponent)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        if not is_almost(self.exponent, 2.0):
            dct["exponent"] = to_engstr(self.exponent)
        return dct


class ValueRangeOperator(PixelRegularizationOperator):
    def input_image_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.value_range_loss(input_repr)
