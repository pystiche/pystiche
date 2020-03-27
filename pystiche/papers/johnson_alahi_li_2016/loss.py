from typing import Any, Union, Optional, Sequence, Dict
import torch
import pystiche.ops.functional as F
from pystiche.enc import Encoder, MultiLayerEncoder
from pystiche.ops import (
    MSEEncodingOperator,
    GramOperator,
    TotalVariationOperator,
    MultiLayerEncodingOperator,
)
from pystiche.loss import PerceptualLoss
from pystiche.misc import warn_deprecation
from .utils import johnson_alahi_li_2016_multi_layer_encoder

__all__ = [
    "johnson_alahi_li_2016_content_loss",
    "johnson_alahi_li_2016_style_loss",
    "johnson_alahi_li_2016_regularization",
    "johnson_alahi_li_2016_perceptual_loss",
]


def get_content_score_weight(instance_norm: bool, style: Optional[str] = None) -> float:
    default_score_weight = 1e0

    if style is None or not instance_norm:
        return default_score_weight

    score_weights = {"la_muse": 0.5, "udnie": 0.5}
    try:
        return score_weights[style]
    except KeyError:
        return default_score_weight


def johnson_alahi_li_2016_content_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    style: Optional[str] = None,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu2_2",
    score_weight: Optional[float] = None,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = johnson_alahi_li_2016_multi_layer_encoder(
            impl_params=impl_params
        )
    encoder = multi_layer_encoder.extract_single_layer_encoder(layer)

    if score_weight is None:
        score_weight = get_content_score_weight(instance_norm, style=style)

    return MSEEncodingOperator(encoder, score_weight=score_weight)


class JohnsonAlahiLi2016GramOperator(GramOperator):
    def __init__(
        self, encoder: Encoder, impl_params: bool = True, **gram_op_kwargs: Any,
    ):
        super().__init__(encoder, **gram_op_kwargs)
        self.normalize_by_num_channels = impl_params

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        gram_matrix = super().enc_to_repr(enc)
        if not self.normalize_by_num_channels:
            return gram_matrix

        num_channels = gram_matrix.size()[-1]
        return gram_matrix / num_channels


def get_style_score_weight(
    impl_params: bool, instance_norm: bool, style: Optional[str] = None
) -> float:
    if style is None or not impl_params:
        return 5.0

    if instance_norm:
        if style == "the_scream":
            return 20.0
        else:
            return 10.0
    else:
        if style == "starry_night":
            return 3.0
        else:
            return 5.0


def johnson_alahi_li_2016_style_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    style: Optional[str] = None,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    score_weight: float = 5e0,
    **gram_op_kwargs,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = johnson_alahi_li_2016_multi_layer_encoder(
            impl_params=impl_params
        )

    if layers is None:
        layers = ("relu1_2", "relu2_2", "relu3_3", "relu4_3")

    if score_weight is None:
        score_weight = get_style_score_weight(impl_params, instance_norm, style=style)

    def get_encoding_op(encoder, layer_weight):
        return JohnsonAlahiLi2016GramOperator(
            encoder, score_weight=layer_weight, **gram_op_kwargs
        )

    return MultiLayerEncodingOperator(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


class JohnsonAlahiLi2016TotalVariationOperator(TotalVariationOperator):
    def __init__(self, **total_variation_op_kwargs):
        super().__init__(**total_variation_op_kwargs)

        self.loss_reduction = "sum"

    def calculate_score(self, input_repr):
        return F.total_variation_loss(
            input_repr, exponent=self.exponent, reduction=self.loss_reduction
        )


def get_regularization_score_weight(
    instance_norm: bool, style: Optional[str] = None
) -> float:
    default_score_weight = 1e-6

    if style is None:
        return default_score_weight

    if instance_norm:
        score_weights = {
            "candy": 1e-4,
            "la_muse": 1e-4,
            "mosaic": 1e-5,
            "feathers": 1e-5,
            "the_scream": 1e-5,
            "udnie": 1e-6,
        }
        try:
            return score_weights[style]
        except KeyError:
            return default_score_weight
    else:
        score_weights = {
            "the_wave": 1e-4,
            "starry_night": 1e-5,
            "la_muse": 1e-5,
            "composition_vii": 1e-6,
        }
        try:
            return score_weights[style]
        except KeyError:
            return default_score_weight


def johnson_alahi_li_2016_regularization(
    instance_norm: bool = True,
    style: Optional[str] = None,
    score_weight: Optional[float] = None,
    **total_variation_op_kwargs: Any,
):
    if score_weight is None:
        score_weight = get_regularization_score_weight(instance_norm, style=style)
    return JohnsonAlahiLi2016TotalVariationOperator(
        score_weight=score_weight, **total_variation_op_kwargs
    )


class JohnsonAlahiLi2016PerceptualLoss(PerceptualLoss):
    def __init__(
        self,
        content_loss: MSEEncodingOperator,
        style_loss: MultiLayerEncodingOperator,
        regularization: TotalVariationOperator,
    ) -> None:
        warn_deprecation(
            "class",
            "JohnsonAlahiLi2016PerceptualLoss",
            "0.4",
            info="It can be replaced by pystiche.loss.PerceptualLoss.",
        )
        super().__init__(content_loss, style_loss, regularization=regularization)


def johnson_alahi_li_2016_perceptual_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    style: Optional[str] = None,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
    total_variation_kwargs: Optional[Dict[str, Any]] = None,
) -> JohnsonAlahiLi2016PerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = johnson_alahi_li_2016_multi_layer_encoder(
            impl_params=impl_params
        )

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss = johnson_alahi_li_2016_content_loss(
        impl_params=impl_params,
        instance_norm=instance_norm,
        style=style,
        multi_layer_encoder=multi_layer_encoder,
        **content_loss_kwargs,
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = johnson_alahi_li_2016_style_loss(
        impl_params=impl_params,
        instance_norm=instance_norm,
        style=style,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    if total_variation_kwargs is None:
        total_variation_kwargs = {}
    regularization = johnson_alahi_li_2016_regularization(
        instance_norm=instance_norm, style=style, **total_variation_kwargs
    )

    return PerceptualLoss(content_loss, style_loss, regularization=regularization)
