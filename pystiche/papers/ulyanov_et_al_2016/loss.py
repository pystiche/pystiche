from typing import Union, Optional, Sequence, Any, Dict
from collections import OrderedDict
import torch
import pystiche
from pystiche.enc import Encoder, MultiLayerEncoder
from pystiche.ops import (
    MSEEncodingOperator,
    GramOperator,
    MultiLayerEncodingOperator,
)
from pystiche.loss import MultiOperatorLoss
from .utils import ulyanov_et_al_2016_multi_layer_encoder


def ulyanov_et_al_2016_content_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu_4_2",
    score_weight=None,
):
    if score_weight is None:
        if impl_params:
            score_weight = 1e0 if instance_norm else 6e-1
        else:
            score_weight = 1e0

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
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    score_weight: float = None,
    **gram_op_kwargs,
):
    if score_weight is None:
        if impl_params:
            if instance_norm:
                score_weight = 1e0
            else:
                score_weight = 1e3 if stylization else 1e0
        else:
            score_weight = 1e0

    if multi_layer_encoder is None:
        multi_layer_encoder = ulyanov_et_al_2016_multi_layer_encoder()

    if layers is None:
        if impl_params:
            layers = (
                ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1")
                if instance_norm
                else ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
            )
        else:
            layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")

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


class UlyanovEtAl2016PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self,
        style_loss: MultiLayerEncodingOperator,
        content_loss: MSEEncodingOperator = None,
        stylization: bool = True,
    ) -> None:

        dict = OrderedDict([("style_loss", style_loss)])
        if stylization:
            dict["content_loss"] = content_loss

        super().__init__(dict)

    def set_content_image(self, image: torch.Tensor):
        self.content_loss.set_target_image(image)

    def set_style_image(self, image: torch.Tensor):
        self.style_loss.set_target_image(image)

    def forward(self, input_image: torch.Tensor) -> pystiche.LossDict:
        for encoder in self._multi_layer_encoders:
            encoder.encode(input_image)

        loss = pystiche.LossDict(
            [(name, op(input_image)) for name, op in self.named_children()]
        )

        for key in loss.keys():
            loss[key] = loss[key] / input_image.size()[0]

        for encoder in self._multi_layer_encoders:
            encoder.clear_cache()

        return loss


def ulyanov_et_al_2016_perceptual_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
) -> UlyanovEtAl2016PerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = ulyanov_et_al_2016_multi_layer_encoder()

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = ulyanov_et_al_2016_style_loss(
        impl_params=impl_params,
        instance_norm=instance_norm,
        stylization=stylization,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    if stylization:
        if content_loss_kwargs is None:
            content_loss_kwargs = {}
        content_loss = ulyanov_et_al_2016_content_loss(
            impl_params=impl_params,
            instance_norm=instance_norm,
            multi_layer_encoder=multi_layer_encoder,
            **content_loss_kwargs,
        )
        return UlyanovEtAl2016PerceptualLoss(
            style_loss, content_loss=content_loss, stylization=stylization
        )
    return UlyanovEtAl2016PerceptualLoss(style_loss, stylization=stylization)
