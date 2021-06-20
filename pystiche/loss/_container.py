from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch

from pystiche import LossDict, enc
from pystiche.misc import verify_str_arg

from ._loss import ComparisonLoss, Loss, RegularizationLoss

__all__ = [
    "LossContainer",
    "SameTypeLossContainer",
    "MultiLayerEncodingLoss",
    "MultiRegionLoss",
    "PerceptualLoss",
]


class LossContainer(Loss):
    def __init__(
        self,
        named_losses: Sequence[Tuple[str, Loss]],
        *,
        input_guide: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        target_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1e0,
    ):
        super().__init__(score_weight=score_weight)
        for name, loss in named_losses:
            self.add_module(name, loss)

        if input_guide is not None:
            self.set_input_guide(input_guide)

        if target_image is not None:
            self.set_target_image(target_image, guide=target_guide)

    def forward(self, input_image: torch.Tensor) -> LossDict:
        return (
            LossDict([(name, loss(input_image)) for name, loss in self._named_losses()])
            * self.score_weight
        )

    def _get_loss_attr(self, attr: str, losses: Optional[Iterable[Loss]] = None) -> Any:
        if losses is None:
            losses = self._losses()

        values = {getattr(loss, attr) for loss in losses}
        if not values:
            return None

        if len(values) > 1:
            raise RuntimeError()

        return values.pop()

    @property
    def encoder(self) -> Optional[enc.Encoder]:
        return cast(Optional[enc.Encoder], self._get_loss_attr("encoder"))

    @property
    def input_guide(self) -> Optional[torch.Tensor]:
        return cast(Optional[torch.Tensor], self._get_loss_attr("input_guide"))

    def _losses_with_target(self) -> Iterator[Union[ComparisonLoss, "LossContainer"]]:
        at_least_one = False
        for loss in self._losses():
            if isinstance(loss, (ComparisonLoss, LossContainer)):
                at_least_one = True
                yield loss

        if not at_least_one:
            raise RuntimeError

    @property
    def target_image(self) -> Optional[torch.Tensor]:
        return cast(
            Optional[torch.Tensor],
            self._get_loss_attr("target_image", self._losses_with_target()),
        )

    @property
    def target_guide(self) -> Optional[torch.Tensor]:
        return cast(
            Optional[torch.Tensor],
            self._get_loss_attr("target_guide", self._losses_with_target()),
        )

    def set_input_guide(self, guide: torch.Tensor) -> None:
        for loss in self._losses():
            loss.set_input_guide(guide)

    def set_target_image(
        self, image: torch.Tensor, *, guide: Optional[torch.Tensor] = None
    ) -> None:
        at_least_one = False
        for loss in self._losses():
            if not isinstance(loss, (ComparisonLoss, LossContainer)):
                continue

            at_least_one = True
            loss.set_target_image(image, guide=guide)

        if not at_least_one:
            raise RuntimeError


class SameTypeLossContainer(LossContainer):
    def __init__(
        self,
        names: Sequence[str],
        loss_fn: Callable[[str, float], Loss],
        *,
        loss_weights: Union[str, Sequence[float]] = "sum",
        input_guide: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        target_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1e0,
    ) -> None:
        op_weights = self._parse_loss_weights(loss_weights, len(names))
        named_losses = [
            (name, loss_fn(name, weight)) for name, weight in zip(names, op_weights)
        ]

        super().__init__(
            named_losses,
            input_guide=input_guide,
            target_image=target_image,
            target_guide=target_guide,
            score_weight=score_weight,
        )

    @staticmethod
    def _parse_loss_weights(
        loss_weights: Union[str, Sequence[float]], num_losses: int
    ) -> Sequence[float]:
        if isinstance(loss_weights, str):
            verify_str_arg(loss_weights, "loss_weights", ("sum", "mean"))
            if loss_weights == "sum":
                return [1.0] * num_losses
            else:  # loss_weights == "mean":
                return [1.0 / num_losses] * num_losses
        else:
            if len(loss_weights) == num_losses:
                return loss_weights

            msg = (
                f"The length of the loss weights and the number of losses do not match: "
                f"{len(loss_weights)} != {num_losses}"
            )
            raise ValueError(msg)


class MultiLayerEncodingLoss(SameTypeLossContainer):
    def __init__(
        self,
        mle: enc.MultiLayerEncoder,
        layers: Sequence[str],
        encoding_loss_fn: Callable[[enc.Encoder, float], Loss],
        *,
        layer_weights: Union[str, Sequence[float]] = "sum",
        input_guide: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        target_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1e0,
    ):
        def loss_fn(layer: str, layer_weight: float) -> Loss:
            encoder = mle.extract_encoder(layer)
            return encoding_loss_fn(encoder, layer_weight)

        super().__init__(
            layers,
            loss_fn,
            loss_weights=layer_weights,
            input_guide=input_guide,
            target_image=target_image,
            target_guide=target_guide,
            score_weight=score_weight,
        )


class MultiRegionLoss(SameTypeLossContainer):
    def __init__(
        self,
        regions: Sequence[str],
        region_loss_fn: Callable[[str, float], Loss],
        *,
        region_weights: Union[str, Sequence[float]] = "sum",
        input_guide: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        target_guide: Optional[torch.Tensor] = None,
        score_weight: float = 1e0,
    ):
        super().__init__(
            regions,
            region_loss_fn,
            loss_weights=region_weights,
            input_guide=input_guide,
            target_image=target_image,
            target_guide=target_guide,
            score_weight=score_weight,
        )

    def set_regional_input_guide(self, region: str, guide: torch.Tensor) -> None:
        r"""Invokes :meth:`~pystiche.ops.Comparison.set_input_guide` on the operator
        of the given ``region``.

        Args:
            region: Region.
            guide: Input guide of shape :math:`1 \times 1 \times H \times W`.
        """
        getattr(self, region).set_input_guide(guide)

    def set_regional_target_image(
        self, region: str, image: torch.Tensor, guide: Optional[torch.Tensor] = None
    ) -> None:
        r"""Invokes :meth:`~pystiche.ops.Comparison.set_target_image` on the operator
        of the given ``region``.

        Args:
            region: Region.
            image: Input guide of shape :math:`B \times C \times H \times W`.
        """
        getattr(self, region).set_target_image(image, guide)


class PerceptualLoss(LossContainer):
    def __init__(
        self,
        content_loss: Union[ComparisonLoss, LossContainer],
        style_loss: Union[ComparisonLoss, LossContainer],
        regularization: Optional[RegularizationLoss] = None,
        *,
        content_image: Optional[torch.Tensor] = None,
        content_guide: Optional[torch.Tensor] = None,
        style_image: Optional[torch.Tensor] = None,
        style_guide: Optional[torch.Tensor] = None,
    ) -> None:
        named_losses: List[Tuple[str, Loss]] = [
            ("content_loss", content_loss),
            ("style_loss", style_loss),
        ]
        if regularization is not None:
            named_losses.append(("regularization", regularization))

        super().__init__(named_losses)
        self.content_loss: Union[ComparisonLoss, LossContainer]
        self.style_loss: Union[ComparisonLoss, LossContainer]

        if content_image is not None:
            self.set_content_image(content_image)

        if content_guide is not None:
            self.set_content_guide(content_guide)

        if style_image is not None:
            self.set_style_image(style_image, style_guide)

    @property
    def content_image(self) -> Optional[torch.Tensor]:
        return self.content_loss.target_image

    def set_content_image(self, image: torch.Tensor) -> None:
        self.content_loss.set_target_image(image)

    def _regional_style_loss(self, region: Optional[str]) -> Loss:
        return self.style_loss if not region else getattr(self.style_loss, region)

    @property
    def content_guide(self) -> Optional[torch.Tensor]:
        return self.style_loss.input_guide

    def regional_content_guide(
        self, region: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        return self._regional_style_loss(region).input_guide

    def set_content_guide(
        self, guide: torch.Tensor, *, region: Optional[str] = None
    ) -> None:
        self._regional_style_loss(region).set_input_guide(guide)

    @property
    def style_image(self) -> Optional[torch.Tensor]:
        return self.style_loss.target_image

    def regional_style_image(
        self, region: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        return cast(
            Union[ComparisonLoss, LossContainer], self._regional_style_loss(region)
        ).target_image

    @property
    def style_guide(self) -> Optional[torch.Tensor]:
        return self.style_loss.target_guide

    def regional_style_guide(
        self, region: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        return cast(
            Union[ComparisonLoss, LossContainer], self._regional_style_loss(region)
        ).target_guide

    def set_style_image(
        self,
        image: torch.Tensor,
        guide: Optional[torch.Tensor] = None,
        *,
        region: Optional[str] = None,
    ) -> None:
        a = cast(
            Union[ComparisonLoss, LossContainer], self._regional_style_loss(region)
        )
        a.set_target_image(image, guide=guide)
        return None
