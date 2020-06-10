from typing import List, Optional, Tuple, Union

import torch

from pystiche.ops import (
    ComparisonOperator,
    Operator,
    OperatorContainer,
    RegularizationOperator,
)

from .multi_op import MultiOperatorLoss

__all__ = ["PerceptualLoss", "GuidedPerceptualLoss"]


class _PerceptualLoss(MultiOperatorLoss):
    content_loss: Union[ComparisonOperator, OperatorContainer]
    style_loss: Union[ComparisonOperator, OperatorContainer]
    regularization: Optional[RegularizationOperator]

    def __init__(
        self,
        content_loss: Union[ComparisonOperator, OperatorContainer],
        style_loss: Union[ComparisonOperator, OperatorContainer],
        regularization: Optional[RegularizationOperator] = None,
        trim: bool = True,
    ) -> None:
        named_ops: List[Tuple[str, Operator]] = [
            ("content_loss", content_loss),
            ("style_loss", style_loss),
        ]

        if regularization is not None:
            named_ops.append(("regularization", regularization))

        super().__init__(named_ops, trim=trim)

    def set_content_image(self, image: torch.Tensor) -> None:
        r"""Set the content image.

        Args:
            image: Content image.
        """
        self.content_loss.set_target_image(image)


class PerceptualLoss(_PerceptualLoss):
    r"""Perceptual loss comprising content and style loss as well as optionally a
    regularization.

    Args:
        content_loss: Content loss.
        style_loss: Style loss.
        regularization: Optional regularization.
        trim: If ``True``, all :class:`~pystiche.enc.MultiLayerEncoder` s associated
            with ``content_loss``, ``style_loss``, or ``regularization`` will be
            :meth:`~pystiche.enc.MultiLayerEncoder.trim` med. Defaults to ``True``.
    """

    def set_style_image(self, image: torch.Tensor) -> None:
        r"""Set the style image.

        Args:
            image: Style image.
        """
        self.style_loss.set_target_image(image)


class GuidedPerceptualLoss(_PerceptualLoss):
    r"""Guided perceptual loss comprising content and guided style loss as well as
    optionally a regularization.

    Args:
        content_loss: Content loss.
        style_loss: Guided style loss.
        regularization: Optional regularization.
        trim: If ``True``, all :class:`~pystiche.enc.MultiLayerEncoder` s associated
            with ``content_loss``, ``style_loss``, or ``regularization`` will be
            :meth:`~pystiche.enc.MultiLayerEncoder.trim` med. Defaults to ``True``.
    """

    def set_style_guide(
        self, region: str, guide: torch.Tensor, recalc_repr: bool = True
    ) -> None:
        r"""Set the style guide for the specified region.

        Args:
            region: Region.
            guide: Style guide.
            recalc_repr: If ``True``, recalculates the style representation. See
                :meth:`pystiche.ops.ComparisonOperator.set_target_guide` for details.
                Defaults to ``True``.
        """
        getattr(self.style_loss, region).set_target_guide(
            guide, recalc_repr=recalc_repr
        )

    def set_style_image(self, region: str, image: torch.Tensor) -> None:
        r"""Set the style image for the specified region.

        Args:
            region: Region.
            image: Style image.
        """
        getattr(self.style_loss, region).set_target_image(image)

    def set_content_guide(self, region: str, guide: torch.Tensor) -> None:
        r"""Set the content guide for the specified region.

        Args:
            region: Region.
            guide: Content guide.
        """
        getattr(self.style_loss, region).set_input_guide(guide)
