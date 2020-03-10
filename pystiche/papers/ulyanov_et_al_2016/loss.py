from typing import Union, Optional, Sequence, Any, Dict
from collections import OrderedDict
import torch
from pystiche.enc import Encoder, MultiLayerEncoder
from pystiche.ops import (
    MSEEncodingOperator,
    GramOperator,
    MultiLayerEncodingOperator,
    TotalVariationOperator,
)
from pystiche.loss import MultiOperatorLoss
from .utils import ulyanov_et_al_2016_multi_layer_encoder


def ulyanov_et_al_2016_content_loss(
    impl_mode: str = "paper",
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu_4_2",
    score_weight=None,
):
    if score_weight is None:
        if impl_mode == "texturenetsv1":
            score_weight = 6e-1
        elif impl_mode == "paper":
            score_weight = 1e0  # FIXME paper only alpha
        elif impl_mode == "master":
            score_weight = 1e0
        else:
            raise NotImplementedError

    if multi_layer_encoder is None:
        multi_layer_encoder = ulyanov_et_al_2016_multi_layer_encoder()
    encoder = multi_layer_encoder[layer]

    return MSEEncodingOperator(encoder, score_weight=score_weight)


class UlyanovEtAl2016GramOperator(GramOperator):
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


def ulyanov_et_al_2016_style_loss(
    impl_mode: str = "paper",
    mode: str = "texture",
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    score_weight: float = None,
    **gram_op_kwargs,
):
    if score_weight is None:
        if impl_mode == "texturenetsv1":
            score_weight = 1e3 if mode == "style" else 1e0
        elif impl_mode == "paper":
            score_weight = 1e0
        elif impl_mode == "master":
            score_weight = 1e0
        else:
            raise NotImplementedError

    if multi_layer_encoder is None:
        multi_layer_encoder = ulyanov_et_al_2016_multi_layer_encoder()

    if layers is None:
        if impl_mode == "texturenetsv1":
            layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
        elif impl_mode == "paper":
            layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
        elif impl_mode == "master":
            layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1")
        else:
            raise NotImplementedError

    def get_encoding_op(encoder, layer_weight):
        return UlyanovEtAl2016GramOperator(
            encoder, score_weight=layer_weight, **gram_op_kwargs
        )

    return MultiLayerEncodingOperator(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


def ulyanov_et_al_2016_regularization(
    impl_mode: str = "paper",
    score_weight: float = None,
    **total_variation_op_kwargs: Any,
):
    if score_weight is None:
        if impl_mode == "texturenetsv1":
            return None
        elif impl_mode == "paper":
            return None
        elif impl_mode == "master":
            score_weight = 0  # FIXME: tvLoss only in master but weight 0
        else:
            raise NotImplementedError
    if score_weight == 0:
        return None
    return TotalVariationOperator(
        score_weight=score_weight, **total_variation_op_kwargs
    )


class UlyanovEtAl2016PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self,
        style_loss: MultiLayerEncodingOperator,
        regularization: TotalVariationOperator = None,
        content_loss: MSEEncodingOperator = None,
        mode: str = "texture",
    ) -> None:

        dict = OrderedDict([("style_loss", style_loss)])
        if mode == "style":
            dict["content_loss"] = content_loss
        if regularization is not None:
            dict["regularization"] = regularization

        super().__init__(dict)

    def set_content_image(self, image: torch.Tensor):
        self.content_loss.set_target_image(image)

    def set_style_image(self, image: torch.Tensor):
        self.style_loss.set_target_image(image)


def ulyanov_et_al_2016_perceptual_loss(
    impl_mode: str = "paper",
    mode: str = "texture",
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
    total_variation_kwargs: Optional[Dict[str, Any]] = None,
) -> UlyanovEtAl2016PerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = ulyanov_et_al_2016_multi_layer_encoder()

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = ulyanov_et_al_2016_style_loss(
        impl_mode=impl_mode,
        mode=mode,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    if total_variation_kwargs is None:
        total_variation_kwargs = {}
    regularization = ulyanov_et_al_2016_regularization(
        impl_mode=impl_mode, **total_variation_kwargs
    )

    if mode == "style":
        if content_loss_kwargs is None:
            content_loss_kwargs = {}
        content_loss = ulyanov_et_al_2016_content_loss(
            impl_mode=impl_mode,
            multi_layer_encoder=multi_layer_encoder,
            **content_loss_kwargs,
        )
        return UlyanovEtAl2016PerceptualLoss(
            style_loss, regularization, content_loss=content_loss, mode=mode
        )
    return UlyanovEtAl2016PerceptualLoss(style_loss, regularization, mode=mode)
