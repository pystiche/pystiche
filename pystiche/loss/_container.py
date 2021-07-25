from collections import OrderedDict
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
    r"""Generic container for :class:`~pystiche.loss.Loss`'es.

    If called with an image, it will be passes it to all immediate losses and returns
    a :class:`pystiche.LossDict` scaled with ``score_weight``.

    Args:
        named_losses: Named immediate losses that will be called if
            :class:`OperatorContainer` is called.
        score_weight: Score weight of the loss. Defaults to ``1.0``.
    """

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
            raise RuntimeError(f"Found more than one value for attribute {attr}.")

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
            raise RuntimeError("No children losses with target found.")

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
        for loss in self._losses_with_target():
            loss.set_target_image(image, guide=guide)


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

            raise ValueError(
                f"The length of the loss weights and the number of losses do not "
                f"match: {len(loss_weights)} != {num_losses}"
            )


class MultiLayerEncodingLoss(SameTypeLossContainer):
    r"""Convenience container for multiple :class:`~pystiche.loss.Loss`'es
    operating on different ``layers`` of the same
    :class:`pystiche.enc.MultiLayerEncoder`.

    Args:
        mle: Multi-layer encoder.
        layers: Layers of the ``mle`` that the children losses operate on.
        encoding_loss_fn: Callable that returns a loss given a
            :class:`pystiche.enc.SingleLayerEncoder` extracted from the ``mle`` and its
            corresponding layer weight.
        layer_weights: Weights passed to ``encoding_loss_fn``. If ``"sum"``, each layer
            weight is set to ``1.0``. If ``"mean"``, each layer weight is set to
            ``1.0 / len(layers)``. If sequence of ``float``s its length has to match
            ``layers``' length. Defaults to ``"mean"``.
        score_weight: Score weight of the loss. Defaults to ``1.0``.

    Examples:

        >>> mle = pystiche.enc.vgg19_multi_layer_encoder()
        >>> layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
        >>> loss = pystiche.loss.MultiLayerEncodingLoss(
        ...     mle,
        ...     layers,
        ...     lambda encoder, layer_weight: pystiche.loss.GramLoss(
        ...         encoder, score_weight=layer_weight
        ...     ),
        ... )
        >>> input = torch.rand(2, 3, 256, 256)
        >>> target = torch.rand(2, 3, 256, 256)
        >>> loss.set_target_image(target)
        >>> score = loss(input)
    """

    def __init__(
        self,
        mle: enc.MultiLayerEncoder,
        layers: Sequence[str],
        encoding_loss_fn: Callable[[enc.Encoder, float], Loss],
        *,
        layer_weights: Union[str, Sequence[float]] = "mean",
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

    def __repr__(self) -> str:
        def build_encoder_repr() -> str:
            layer_loss = next(self._losses())
            mle = cast(enc.SingleLayerEncoder, layer_loss.encoder).multi_layer_encoder
            name = mle.__class__.__name__
            properties = mle.properties()
            named_children = ()
            return self._build_repr(
                name=name, properties=properties, named_children=named_children
            )

        def build_layer_repr(loss: Loss) -> str:
            properties = loss.properties()
            return loss._build_repr(properties=properties, named_children=())

        properties = OrderedDict()
        properties["encoder"] = build_encoder_repr()
        properties.update(self.properties())

        named_children = [
            (name, build_layer_repr(loss)) for name, loss in self._named_losses()
        ]

        return self._build_repr(properties=properties, named_children=named_children)


class MultiRegionLoss(SameTypeLossContainer):
    r"""Convenience container for multiple :class:`~pystiche.loss.Loss`'es
    operating in different ``regions``.

    Args:
        regions: Regions.
        region_loss_fn: Callable that returns a children loss given a region and
            its corresponding weight.
        region_weights: Weights passed to ``region_loss_fn``. If ``"sum"``, each region
            weight is set to ``1.0``. If ``"mean"``, each region weight is set to
            ``1.0 / len(layers)``. If sequence of ``float``s its length has to match
            ``regions``' length. Defaults to ``"mean"``.
        score_weight: Score weight of the loss. Defaults to ``1.0``.

    Examples:
        >>> mle = pystiche.enc.vgg19_multi_layer_encoder()
        >>> layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
        >>> def encoding_loss_fn(encoder, layer_weight):
        ...     return pystiche.loss.GramLoss(encoder, score_weight=layer_weight)
        >>> regions = ("sky", "landscape")
        >>> def region_loss_fn(region, region_weight):
        ...     return pystiche.loss.MultiLayerEncodingLoss(
        ...         mle,
        ...         layers,
        ...         encoding_loss_fn,
        ...         score_weight=region_weight,
        ...     )
        >>> loss = pystiche.loss.MultiRegionLoss(regions, region_loss_fn)
        >>> loss.set_regional_target_image("sky", torch.rand(2, 3, 256, 256))
        >>> loss.set_regional_target_image("landscape", torch.rand(2, 3, 256, 256))
        >>> input = torch.rand(2, 3, 256, 256)
        >>> score = loss(input)
    """

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
            guide:
        """
        getattr(self, region).set_target_image(image, guide=guide)


class PerceptualLoss(LossContainer):
    r"""Perceptual loss comprising content and style loss as well as optionally a
    regularization.

    Args:
        content_loss: Content loss.
        style_loss: Guided style loss.
        regularization: Optional regularization.
        content_image: Content image applied as target image to the ``content_loss``.
        content_guide: Content guide applied as input guide to the ``style_loss``.
        style_image: Style image applied as target image to ``style_loss``.
        style_guide: Style guide applied as target image to ``style_loss``.
    """

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
            self.set_style_image(style_image, guide=style_guide)

    @property
    def content_image(self) -> Optional[torch.Tensor]:
        r"""Content guide."""
        return self.content_loss.target_image

    def set_content_image(self, image: torch.Tensor) -> None:
        r"""Sets the content image.

        Args:
            image: Content image.
        """
        self.content_loss.set_target_image(image)

    def _regional_style_loss(
        self, region: Optional[str]
    ) -> Union[ComparisonLoss, LossContainer]:
        return self.style_loss if not region else getattr(self.style_loss, region)

    @property
    def content_guide(self) -> Optional[torch.Tensor]:
        r"""Content guide."""
        return self.style_loss.input_guide

    def regional_content_guide(
        self, region: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        r"""Regional content guide.

        Args:
            region: Region to get the content guide from.
        """
        return self._regional_style_loss(region).input_guide

    def set_content_guide(
        self, guide: torch.Tensor, *, region: Optional[str] = None
    ) -> None:
        r"""Sets the content guide.

        Args:
            guide: Content guide.
            region: Optional region to set the guide for. If omitted, the guide will be
                applied to all regions.
        """
        self._regional_style_loss(region).set_input_guide(guide)

    @property
    def style_image(self) -> Optional[torch.Tensor]:
        r"""Style image."""
        return self.style_loss.target_image

    def regional_style_image(
        self, region: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        r"""Regional style image.

        Args:
            region: Region to get the style image from.
        """
        return self._regional_style_loss(region).target_image

    @property
    def style_guide(self) -> Optional[torch.Tensor]:
        r"""Style guide."""
        return self.style_loss.target_guide

    def regional_style_guide(
        self, region: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        r"""Regional style guide.

        Args:
            region: Region to get the style guide from.
        """
        return self._regional_style_loss(region).target_guide

    def set_style_image(
        self,
        image: torch.Tensor,
        *,
        guide: Optional[torch.Tensor] = None,
        region: Optional[str] = None,
    ) -> None:
        r"""Sets the style image and guide.

        Args:
            image: Style image.
            guide: Style guide.
            region: Optional region to set the image and guide for. If omitted, the
                image and guide will be applied to all regions.
        """
        self._regional_style_loss(region).set_target_image(image, guide=guide)
