from typing import Any, Union, Optional, Sequence, Dict, Callable
from collections import OrderedDict
import torch
from torch.nn.functional import mse_loss
import pystiche
from pystiche.enc import MultiLayerEncoder, Encoder
from pystiche.ops import (
    EncodingOperator,
    MSEEncodingOperator,
    GramOperator,
    MultiLayerEncodingOperator,
)
from pystiche.loss import MultiOperatorLoss
from .utils import gatys_ecker_bethge_2015_multi_layer_encoder


__all__ = [
    "GatysEckerBethge2015MSEEncodingOperator",
    "gatys_ecker_bethge_2015_content_loss",
    "GatysEckerBethge2015StyleLoss",
    "gatys_ecker_bethge_2015_style_loss",
    "GatysEckerBethge2015PerceptualLoss",
    "gatys_ecker_bethge_2015_perceptual_loss",
]


class GatysEckerBethge2015MSEEncodingOperator(MSEEncodingOperator):
    def __init__(
        self, encoder: Encoder, impl_params: bool = True, score_weight: float = 1e0
    ):
        super().__init__(encoder, score_weight=score_weight)

        self.score_correction_factor = 1.0 if impl_params else 1.0 / 2.0
        self.loss_reduction = "mean" if impl_params else "sum"

    def calculate_score(self, input_repr, target_repr, ctx):
        score = mse_loss(input_repr, target_repr, reduction=self.loss_reduction)
        return score * self.score_correction_factor


def gatys_ecker_bethge_2015_content_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu_4_2",
    score_weight: float = 1e0,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_ecker_bethge_2015_multi_layer_encoder()
    encoder = multi_layer_encoder[layer]

    return GatysEckerBethge2015MSEEncodingOperator(
        encoder, impl_params=impl_params, score_weight=score_weight
    )


class GatysEckerBethge2015StyleLoss(MultiLayerEncodingOperator):
    def __init__(
        self,
        multi_layer_encoder: MultiLayerEncoder,
        layers: Sequence[str],
        get_encoding_op: Callable[[Encoder, float], EncodingOperator],
        impl_params: bool = True,
        layer_weights: Optional[Union[str, Sequence[float]]] = None,
        score_weight: float = 1e0,
    ):
        if layer_weights is None:
            if impl_params:
                layer_weights = self.get_default_layer_weights(
                    multi_layer_encoder, layers
                )
            else:
                layer_weights = "mean"

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
        score = super().process_input_image(input_image)
        return score * self.score_correction_factor


def gatys_ecker_bethge_2015_style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Optional[Union[str, Sequence[float]]] = None,
    score_weight: float = 1e3,
    **gram_loss_kwargs: Any,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_ecker_bethge_2015_multi_layer_encoder()

    if layers is None:
        layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")

    def get_encoding_op(encoder, layer_weight):
        return GramOperator(encoder, score_weight=layer_weight, **gram_loss_kwargs)

    return GatysEckerBethge2015StyleLoss(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        impl_params=impl_params,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


class GatysEckerBethge2015PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self,
        content_loss: GatysEckerBethge2015MSEEncodingOperator,
        style_loss: GatysEckerBethge2015StyleLoss,
    ):
        super().__init__(
            OrderedDict([("content_loss", content_loss), ("style_loss", style_loss)])
        )

    def set_content_image(self, image: torch.Tensor):
        self.content_loss.set_target_image(image)

    def set_style_image(self, image: torch.Tensor):
        self.style_loss.set_target_image(image)


def gatys_ecker_bethge_2015_perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_ecker_bethge_2015_multi_layer_encoder(
            impl_params=impl_params
        )

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss = gatys_ecker_bethge_2015_content_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **content_loss_kwargs,
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = gatys_ecker_bethge_2015_style_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    return GatysEckerBethge2015PerceptualLoss(content_loss, style_loss)
