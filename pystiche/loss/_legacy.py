# type: ignore

import warnings
from types import TracebackType
from typing import Any, Iterator, Optional, Sequence, Tuple, Type

import torch
from torch import nn

import pystiche
from pystiche import enc
from pystiche.misc import build_deprecation_message

from ._container import LossContainer, PerceptualLoss
from ._loss import Loss

__all__ = ["MLEHandler", "MultiOperatorLoss", "GuidedPerceptualLoss"]


class MLEHandler(pystiche.ComplexObject):
    def __init__(self, criterion: nn.Module) -> None:
        msg = build_deprecation_message("The class MLEHandler", "1.0")
        warnings.warn(msg)
        self.multi_layer_encoders = {
            loss.encoder.multi_layer_encoder
            for loss in criterion.modules()
            if isinstance(loss, Loss)
            and not isinstance(loss, LossContainer)
            and isinstance(loss.encoder, enc.SingleLayerEncoder)
        }

    def encode(self, input_image: torch.Tensor) -> None:
        msg = build_deprecation_message(
            "The method 'encode'",
            "1.0",
            info=(
                "It is no longer needed to pre-encode the input. "
                "See https://github.com/pmeier/pystiche/issues/435 for details"
            ),
        )
        warnings.warn(msg)

    def clear_cache(self) -> None:
        for mle in self.multi_layer_encoders:
            mle.clear_cache()

    def empty_storage(self) -> None:
        msg = build_deprecation_message(
            "The method 'empty_storage'", "1.0", info="It was renamed to 'clear_cache'."
        )
        warnings.warn(msg)
        self.clear_cache()

    def trim(self) -> None:
        for mle in self.multi_layer_encoders:
            mle.trim()

    def __call__(self, input_image: torch.Tensor) -> "MLEHandler":
        return self

    def __enter__(self) -> None:
        pass

    def __exit__(
        self, exc_type: Type[Exception], exc_val: Exception, exc_tb: TracebackType
    ) -> None:
        for encoder in self.multi_layer_encoders:
            encoder.clear_cache()

    def _named_children(self) -> Iterator[Tuple[str, enc.MultiLayerEncoder]]:
        return ((str(idx), mle) for idx, mle in enumerate(self.multi_layer_encoders))


class MultiOperatorLoss(pystiche.Module):
    r"""Generic loss for multiple :class:`~pystiche.Loss` s. If called with an
    image it is passed to all immediate children operators and the
    results are returned as a :class:`pystiche.LossDict`. For that each
    :class:`pystiche.enc.MultiLayerEncoder` is only hit once, even if it is associated
    with multiple of the called operators.

    Args:
        named_ops: Named children operators.
        trim: If ``True``, all :class:`~pystiche.enc.MultiLayerEncoder` s associated
            with ``content_loss``, ``style_loss``, or ``regularization`` will be
            :meth:`~pystiche.enc.MultiLayerEncoder.trim` med. Defaults to ``True``.
    """

    def __init__(
        self, named_ops: Sequence[Tuple[str, Loss]], trim: bool = True
    ) -> None:
        msg = build_deprecation_message("The class MultiOperatorLoss", "1.0")
        warnings.warn(msg)
        super().__init__(named_children=named_ops)
        self._mle_handler = MLEHandler(self)

        if trim:
            self._mle_handler.trim()

    def named_operators(self, recurse: bool = False) -> Iterator[Tuple[str, Loss]]:
        iterator = self.named_modules() if recurse else self.named_children()
        for name, child in iterator:
            if isinstance(child, Loss):
                yield name, child

    def operators(self, recurse: bool = False) -> Iterator[Loss]:
        for _, op in self.named_operators(recurse=recurse):
            yield op

    def forward(self, input_image: torch.Tensor) -> pystiche.LossDict:
        with self._mle_handler(input_image):
            return pystiche.LossDict(
                [(name, op(input_image)) for name, op in self.named_children()]
            )


class GuidedPerceptualLoss(PerceptualLoss):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        msg = build_deprecation_message(
            "The class GuidedPerceptualLoss",
            "1.0",
            info="pystiche.loss.PerceptualLoss now also handles guided inputs and targets",
        )
        warnings.warn(msg)
        super().__init__(*args, **kwargs)

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

    def set_style_image(
        self,
        region: str,
        image: torch.Tensor,
        *,
        guide: Optional[torch.Tensor] = None,
        _recalc_repr: bool = True,
    ) -> None:
        r"""Set the style image for the specified region.
        Args:
            region: Region.
            image: Style image.
        """
        if guide is not None and image is not None:
            super().set_style_image(image, region=region, guide=guide)
        elif image is None:
            if _recalc_repr and self.style_image is not None:
                super().set_style_image(self.style_image, guide=guide, region=region)
            else:
                self.register_buffer("_target_guide", guide, persistent=False)
        elif guide is None:
            guide = self.regional_style_guide(region)
            super().set_style_image(image, region=region, guide=guide)

    def set_content_guide(self, region: str, guide: torch.Tensor) -> None:
        r"""Set the content guide for the specified region.
        Args:
            region: Region.
            guide: Content guide.
        """
        super().set_content_guide(guide, region=region)
