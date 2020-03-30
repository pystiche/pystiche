from collections import OrderedDict
from typing import Optional
import torch
from pystiche.ops import ComparisonOperator, RegularizationOperator
from .multi_op import MultiOperatorLoss

__all__ = ["PerceptualLoss", "GuidedPerceptualLoss"]


class _PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self,
        content_loss: ComparisonOperator,
        style_loss: ComparisonOperator,
        regularization: Optional[RegularizationOperator] = None,
        trim: bool = True,
    ) -> None:
        ops = [("content_loss", content_loss), ("style_loss", style_loss)]
        if regularization is not None:
            ops.append(("regularization", regularization))
        super().__init__(OrderedDict(ops), trim=trim)

    def set_content_image(self, image: torch.Tensor) -> None:
        self.content_loss.set_target_image(image)


class PerceptualLoss(_PerceptualLoss):
    def set_style_image(self, image: torch.Tensor) -> None:
        self.style_loss.set_target_image(image)


class GuidedPerceptualLoss(_PerceptualLoss):
    def set_style_image(self, region: str, image: torch.Tensor) -> None:
        getattr(self.style_loss, region).set_target_image(image)

    def set_content_guide(self, region: str, guide: torch.Tensor) -> None:
        getattr(self.style_loss, region).set_input_guide(guide)

    def set_style_guide(self, region: str, guide: torch.Tensor) -> None:
        getattr(self.style_loss, region).set_target_guide(guide)
