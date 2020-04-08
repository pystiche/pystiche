from collections import OrderedDict
from typing import Optional

import torch

from pystiche.ops import ComparisonOperator, RegularizationOperator

from .multi_op import MultiOperatorLoss

__all__ = ["PerceptualLoss", "GuidedPerceptualLoss"]


class _PerceptualLoss(MultiOperatorLoss):
    _CONTENT_LOSS_ID = "content_loss"
    _STYLE_LOSS_ID = "style_loss"
    _REGULARIZATION_ID = "regularization"

    def __init__(
        self,
        content_loss: Optional[ComparisonOperator] = None,
        style_loss: Optional[ComparisonOperator] = None,
        regularization: Optional[RegularizationOperator] = None,
        trim: bool = True,
    ) -> None:
        ops = []
        if content_loss is not None:
            ops.append((self._CONTENT_LOSS_ID, content_loss))
        if style_loss is not None:
            ops.append((self._STYLE_LOSS_ID, style_loss))

        if not ops:
            msg = "PerceptualLoss requires at least content_loss or style_loss."
            raise RuntimeError(msg)

        if regularization is not None:
            ops.append((self._REGULARIZATION_ID, regularization))

        super().__init__(OrderedDict(ops), trim=trim)

    @property
    def has_content_loss(self) -> bool:
        return self._CONTENT_LOSS_ID in self._modules

    @property
    def has_style_loss(self) -> bool:
        return self._STYLE_LOSS_ID in self._modules

    @property
    def has_regularization(self) -> bool:
        return self._REGULARIZATION_ID in self._modules

    def _verify_has_content_loss(self):
        if not self.has_content_loss:
            msg = (
                "This instance of PerceptualLoss has no content_loss. If you need a "
                "content_loss construct your instance with "
                "PerceptualLoss(content_loss=..., ...)."
            )
            raise RuntimeError(msg)

    def _verfiy_has_style_loss(self):
        if not self.has_style_loss:
            msg = (
                "This instance of PerceptualLoss has no style_loss. If you need a "
                "style_loss construct your instance with "
                "PerceptualLoss(style_loss=..., ...)."
            )
            raise RuntimeError(msg)

    def set_content_image(self, image: torch.Tensor) -> None:
        self._verify_has_content_loss()
        self.content_loss.set_target_image(image)


class PerceptualLoss(_PerceptualLoss):
    def set_style_image(self, image: torch.Tensor) -> None:
        self._verfiy_has_style_loss()
        self.style_loss.set_target_image(image)


class GuidedPerceptualLoss(_PerceptualLoss):
    def set_style_image(self, region: str, image: torch.Tensor) -> None:
        self._verfiy_has_style_loss()
        getattr(self.style_loss, region).set_target_image(image)

    def set_content_guide(self, region: str, guide: torch.Tensor) -> None:
        self._verfiy_has_style_loss()
        getattr(self.style_loss, region).set_input_guide(guide)

    def set_style_guide(self, region: str, guide: torch.Tensor) -> None:
        self._verfiy_has_style_loss()
        getattr(self.style_loss, region).set_target_guide(guide)
