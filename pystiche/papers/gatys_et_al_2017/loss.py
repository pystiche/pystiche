from typing import Any, Union, Optional, Sequence, Dict, Callable
from collections import OrderedDict
import torch
import pystiche
from pystiche.enc import MultiLayerEncoder, Encoder
from pystiche.ops import (
    EncodingOperator,
    EncodingComparisonGuidance,
    MSEEncodingOperator,
    GramOperator,
    MultiLayerEncodingOperator,
    MultiRegionOperator,
)
from pystiche.loss import MultiOperatorLoss
from .utils import gatys_et_al_2017_multi_layer_encoder

__all__ = [
    "gatys_et_al_2017_content_loss",
    "GatysEtAl2017StyleLoss",
    "gatys_et_al_2017_style_loss",
    "gatys_et_al_2017_guided_style_loss",
    "GatysEtAl2017PerceptualLoss",
    "gatys_et_al_2017_perceptual_loss",
    "GatysEtAl2017GuidedPerceptualLoss",
    "gatys_et_al_2017_guided_perceptual_loss",
]


def gatys_et_al_2017_content_loss(
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu_4_2",
    score_weight=1e0,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()
    encoder = multi_layer_encoder[layer]

    return MSEEncodingOperator(encoder, score_weight=score_weight)


class GatysEtAl2017StyleLoss(MultiLayerEncodingOperator):
    def __init__(
        self,
        multi_layer_encoder: MultiLayerEncoder,
        layers: Sequence[str],
        get_encoding_op: Callable[[Encoder, float], EncodingOperator],
        impl_params: bool = True,
        layer_weights: Optional[Union[str, Sequence[float]]] = None,
        score_weight: float = 1e0,
    ) -> None:
        if layer_weights is None:
            layer_weights = self.get_default_layer_weights(multi_layer_encoder, layers)

        super().__init__(
            multi_layer_encoder,
            layers,
            get_encoding_op,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )

        self.score_correction_factor = 1.0 if impl_params else 1.0 / 4.0

    @staticmethod
    def get_default_layer_weights(
        multi_layer_encoder: MultiLayerEncoder, layers: Sequence[str]
    ) -> Sequence[float]:
        nums_channels = []
        for layer in layers:
            module = multi_layer_encoder._modules[layer.replace("relu", "conv")]
            nums_channels.append(module.out_channels)
        return [1.0 / num_channels ** 2.0 for num_channels in nums_channels]

    def process_input_image(self, input_image: torch.Tensor) -> pystiche.LossDict:
        return super().process_input_image(input_image) * self.score_correction_factor


def gatys_et_al_2017_style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Optional[Union[str, Sequence[float]]] = None,
    score_weight: float = 1e3,
    **gram_op_kwargs: Any,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

    if layers is None:
        layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")

    def get_encoding_op(encoder, layer_weight):
        return GramOperator(encoder, score_weight=layer_weight, **gram_op_kwargs)

    return GatysEtAl2017StyleLoss(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        impl_params=impl_params,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


class GuidedGramOperator(EncodingComparisonGuidance, GramOperator):
    pass


def gatys_et_al_2017_guided_style_loss(
    regions: Sequence[str],
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    region_weights: Union[str, Sequence[float]] = "sum",
    layer_weights: Optional[Union[str, Sequence[float]]] = None,
    score_weight: float = 1e3,
    **gram_op_kwargs: Any,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

    if layers is None:
        layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")

    def get_encoding_op(encoder, layer_weight):
        return GuidedGramOperator(encoder, score_weight=layer_weight, **gram_op_kwargs)

    def get_region_op(region, region_weight):
        return GatysEtAl2017StyleLoss(
            multi_layer_encoder,
            layers,
            get_encoding_op,
            impl_params=impl_params,
            layer_weights=layer_weights,
            score_weight=region_weight,
        )

    return MultiRegionOperator(
        regions, get_region_op, region_weights=region_weights, score_weight=score_weight
    )


class _GatysEtAl2017PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self, content_loss: MSEEncodingOperator, style_loss: GatysEtAl2017StyleLoss,
    ):
        super().__init__(
            OrderedDict([("content_loss", content_loss), ("style_loss", style_loss)])
        )

    def set_content_image(self, image: torch.Tensor):
        self.content_loss.set_target_image(image)


class GatysEtAl2017PerceptualLoss(_GatysEtAl2017PerceptualLoss):
    def __init__(
        self, content_loss: MSEEncodingOperator, style_loss: GatysEtAl2017StyleLoss,
    ):
        super().__init__(content_loss, style_loss)

    def set_style_image(self, image: torch.Tensor):
        self.style_loss.set_target_image(image)


def gatys_et_al_2017_perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss = gatys_et_al_2017_content_loss(
        multi_layer_encoder=multi_layer_encoder, **content_loss_kwargs
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = gatys_et_al_2017_style_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    return GatysEtAl2017PerceptualLoss(content_loss, style_loss)


class GatysEtAl2017GuidedPerceptualLoss(_GatysEtAl2017PerceptualLoss):
    def __init__(
        self, content_loss: MSEEncodingOperator, style_loss: GatysEtAl2017StyleLoss,
    ):

        super().__init__(content_loss, style_loss)

    def set_style_guide(self, region: str, guide: torch.Tensor):
        self.style_loss.set_target_guide(region, guide)

    def set_style_image(self, region: str, image: torch.Tensor):
        self.style_loss.set_target_image(region, image)

    def set_content_guide(self, region: str, guide: torch.Tensor):
        self.style_loss.set_input_guide(region, guide)


def gatys_et_al_2017_guided_perceptual_loss(
    regions: Sequence[str],
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss = gatys_et_al_2017_content_loss(
        multi_layer_encoder=multi_layer_encoder, **content_loss_kwargs
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = gatys_et_al_2017_guided_style_loss(
        regions,
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    return GatysEtAl2017GuidedPerceptualLoss(content_loss, style_loss)
