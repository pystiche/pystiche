from typing import Any, Dict

import torch

from pystiche.misc import is_almost, to_engstr

from . import functional as F
from .op import PixelRegularizationOperator

__all__ = ["TotalVariationOperator", "ValueRangeOperator"]


class TotalVariationOperator(PixelRegularizationOperator):
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
