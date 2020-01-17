import torch
from ._ops import RegularizationOperator, EncodingRegularizationOperator
from . import functional as F


__all__ = ["TotalVariationRegularizationOperator", "ValueRangeRegularizationOperator",
           "TotalVariationEncodingRegularizationOperator"]


class TotalVariationRegularizationOperator(RegularizationOperator):
    def __init__(
        self,
        exponent: float = 2.0,
        score_weight : float = 1.0,
    ):
        super().__init__(score_weight=score_weight)
        self.exponent = exponent

    def input_image_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.total_variation_loss(input_repr, exponent=self.exponent)

    # def _descriptions(self) -> Dict[str, Any]:
    #     dct = super()._descriptions()
    #     dct["Exponent"] = to_engstr(self.exponent)
    #     return dct


class TotalVariationEncodingRegularizationOperator(EncodingRegularizationOperator):
    # FIXME remove, just for testing
    def __init__(
        self,
        encoder,
        layer,
        exponent: float = 2.0,
        score_weight : float = 1.0,
    ):
        super().__init__(encoder, layer, score_weight=score_weight)
        self.exponent = exponent

    def input_enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return enc

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.total_variation_loss(input_repr, exponent=self.exponent)


class ValueRangeRegularizationOperator(RegularizationOperator):
    def input_image_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.value_range_loss(input_repr)
