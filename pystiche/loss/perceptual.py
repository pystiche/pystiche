from typing import List, Optional, Tuple, cast

import torch

from pystiche.ops import ComparisonOperator, Operator, RegularizationOperator

from .multi_op import MultiOperatorLoss

__all__ = ["PerceptualLoss", "GuidedPerceptualLoss"]


class _PerceptualLoss(MultiOperatorLoss):
    content_loss: Optional[ComparisonOperator]
    style_loss: Optional[ComparisonOperator]
    regularization: Optional[RegularizationOperator]

    def __init__(
        self,
        content_loss: Optional[ComparisonOperator] = None,
        style_loss: Optional[ComparisonOperator] = None,
        regularization: Optional[RegularizationOperator] = None,
        trim: bool = True,
    ) -> None:
        named_ops: List[Tuple[str, Operator]] = []
        if content_loss is not None:
            named_ops.append(("content_loss", content_loss))
        if style_loss is not None:
            named_ops.append(("style_loss", style_loss))

        if not named_ops:
            msg = "PerceptualLoss requires at least content_loss or style_loss."
            raise RuntimeError(msg)

        if regularization is not None:
            named_ops.append(("regularization", regularization))

        super().__init__(named_ops, trim=trim)

    @property
    def has_content_loss(self) -> bool:
        return "content_loss" in self._modules

    @property
    def has_style_loss(self) -> bool:
        return "style_loss" in self._modules

    @property
    def has_regularization(self) -> bool:
        return "regularization" in self._modules

    def _verify_has_content_loss(self) -> None:
        if not self.has_content_loss:
            msg = (
                "This instance of PerceptualLoss has no content_loss. If you need a "
                "content_loss construct your instance with "
                "PerceptualLoss(content_loss=..., ...)."
            )
            raise RuntimeError(msg)

    def _verfiy_has_style_loss(self) -> None:
        if not self.has_style_loss:
            msg = (
                "This instance of PerceptualLoss has no style_loss. If you need a "
                "style_loss construct your instance with "
                "PerceptualLoss(style_loss=..., ...)."
            )
            raise RuntimeError(msg)

    def set_content_image(self, image: torch.Tensor) -> None:
        self._verify_has_content_loss()
        cast(ComparisonOperator, self.content_loss).set_target_image(image)


class PerceptualLoss(_PerceptualLoss):
    def set_style_image(self, image: torch.Tensor) -> None:
        self._verfiy_has_style_loss()
        cast(ComparisonOperator, self.style_loss).set_target_image(image)


class GuidedPerceptualLoss(_PerceptualLoss):
    def set_style_guide(
        self, region: str, guide: torch.Tensor, recalc_repr: bool = True
    ) -> None:
        self._verfiy_has_style_loss()
        getattr(self.style_loss, region).set_target_guide(
            guide, recalc_repr=recalc_repr
        )

    def set_style_image(self, region: str, image: torch.Tensor) -> None:
        self._verfiy_has_style_loss()
        getattr(self.style_loss, region).set_target_image(image)

    def set_content_guide(self, region: str, guide: torch.Tensor) -> None:
        self._verfiy_has_style_loss()
        getattr(self.style_loss, region).set_input_guide(guide)
