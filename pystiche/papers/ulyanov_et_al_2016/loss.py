from typing import Union, Optional, Sequence, Any, Dict
from collections import OrderedDict
import torch
from pystiche.enc import MultiLayerEncoder
from pystiche.ops import (
    MSEEncodingOperator,
    GramOperator,
    MultiLayerEncodingOperator,
    TotalVariationOperator,
)
from pystiche.loss import MultiOperatorLoss
from .utils import ulyanov_et_al_2016_multi_layer_encoder


def ulyanov_et_al_2016_content_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu_4_2",
    score_weight=1e0,
):
    score_weight = 1e0 if impl_params else score_weight  # FIXME: paper only alpha
    if multi_layer_encoder is None:
        multi_layer_encoder = ulyanov_et_al_2016_multi_layer_encoder()
    encoder = multi_layer_encoder[layer]

    return MSEEncodingOperator(encoder, score_weight=score_weight)


def ulyanov_et_al_2016_style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    score_weight: float = 1e0,
    **gram_op_kwargs,
):
    score_weight = 1e0 if impl_params else score_weight
    if multi_layer_encoder is None:
        multi_layer_encoder = ulyanov_et_al_2016_multi_layer_encoder()

    if layers is None:
        layers = (
            ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1")
            if impl_params
            else ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
        )

    def get_encoding_op(encoder, layer_weight):
        return GramOperator(encoder, score_weight=layer_weight, **gram_op_kwargs)

    return MultiLayerEncodingOperator(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


def ulyanov_et_al_2016_regularization(
    impl_params: bool = True,
    score_weight: float = 1e-6,
    **total_variation_op_kwargs: Any,
):
    score_weight = 0 if impl_params else score_weight  # TODO: right score weight
    return TotalVariationOperator(
        score_weight=score_weight, **total_variation_op_kwargs
    )


class UlyanovEtAl2016PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self,
        content_loss: MSEEncodingOperator,
        style_loss: MultiLayerEncodingOperator,
        regularization: TotalVariationOperator,
    ) -> None:
        super().__init__(
            OrderedDict(
                [
                    ("content_loss", content_loss),
                    ("style_loss", style_loss),
                    ("regularization", regularization),
                ]
            )
        )

    def set_content_image(self, image: torch.Tensor):
        self.content_loss.set_target_image(image)

    def set_style_image(self, image: torch.Tensor):
        self.style_loss.set_target_image(image)


def ulyanov_et_al_2016_perceptual_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    style: Optional[str] = None,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
    total_variation_kwargs: Optional[Dict[str, Any]] = None,
) -> UlyanovEtAl2016PerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = ulyanov_et_al_2016_multi_layer_encoder()

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss = ulyanov_et_al_2016_content_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **content_loss_kwargs,
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = ulyanov_et_al_2016_style_loss(
        impl_params=impl_params,
        instance_norm=instance_norm,
        style=style,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    if total_variation_kwargs is None:
        total_variation_kwargs = {}
    regularization = ulyanov_et_al_2016_regularization(
        instance_norm=instance_norm, style=style, **total_variation_kwargs
    )

    return UlyanovEtAl2016PerceptualLoss(content_loss, style_loss, regularization)
