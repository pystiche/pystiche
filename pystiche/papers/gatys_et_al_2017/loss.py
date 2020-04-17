import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch

import pystiche
from pystiche.enc import Encoder, MultiLayerEncoder
from pystiche.loss import GuidedPerceptualLoss, PerceptualLoss
from pystiche.misc import build_deprecation_message
from pystiche.ops import (
    Comparison,
    EncodingOperator,
    GramOperator,
    MSEEncodingOperator,
    MultiLayerEncodingOperator,
    MultiRegionOperator,
)

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
    layer: str = "relu4_2",
    score_weight=1e0,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()
    encoder = multi_layer_encoder.extract_single_layer_encoder(layer)

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
            cls=Comparison(),
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
        layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")

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

    def get_region_op(region, region_weight):
        return gatys_et_al_2017_style_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            layers=layers,
            layer_weights=layer_weights,
            score_weight=region_weight,
            **gram_op_kwargs,
        )

    return MultiRegionOperator(
        regions, get_region_op, region_weights=region_weights, score_weight=score_weight
    )


class GatysEtAl2017PerceptualLoss(PerceptualLoss):
    def __init__(
        self, content_loss: MSEEncodingOperator, style_loss: GatysEtAl2017StyleLoss,
    ):
        msg = build_deprecation_message(
            "The class GatysEtAl2017PerceptualLoss",
            "0.4.0",
            info="It can be replaced by pystiche.loss.PerceptualLoss.",
        )
        warnings.warn(msg)
        super().__init__(content_loss, style_loss)


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

    return PerceptualLoss(content_loss, style_loss)


class GatysEtAl2017GuidedPerceptualLoss(GuidedPerceptualLoss):
    def __init__(
        self, content_loss: MSEEncodingOperator, style_loss: GatysEtAl2017StyleLoss,
    ):
        msg = build_deprecation_message(
            "The class GatysEtAl2017GuidedPerceptualLoss",
            "0.4.0",
            info="It can be replaced by pystiche.loss.PerceptualLoss.",
        )
        warnings.warn(msg)
        super().__init__(content_loss, style_loss)


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

    return GuidedPerceptualLoss(content_loss, style_loss)
