from typing import Any, Dict
import torch
from pystiche.typing import Numeric
from pystiche.misc import to_engstr
from pystiche.nst import functional as F
from ._base import PixelRegularizationOperator

__all__ = [
    "TotalVariationPixelRegularizationOperator",
    "ValueRangePixelRegularizationOperator",
]


class TotalVariationPixelRegularizationOperator(PixelRegularizationOperator):
    def __init__(
        self,
        name: str = "Total variation pixel regularization operator",
        exponent: Numeric = 2.0,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.exponent = exponent

    def _input_image_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def _calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.total_variation_loss(input_repr, exponent=self.exponent)

    def _descriptions(self) -> Dict[str, Any]:
        dct = super()._descriptions()
        dct["Exponent"] = to_engstr(self.exponent)
        return dct


class ValueRangePixelRegularizationOperator(PixelRegularizationOperator):
    def __init__(
        self, name: str = "value range pixel regularization operator", **kwargs
    ):
        super().__init__(name, **kwargs)

    def _input_image_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def _calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.value_range_loss(input_repr)
