from typing import List, Optional, Tuple
from collections import OrderedDict

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

import pystiche
from pystiche.papers.sanakoyeu_et_al_2018.modules import (
    SanakoyeuEtAl2018TransformerBlock,
)
from pystiche.ops import RegularizationOperator
from pystiche.papers.sanakoyeu_et_al_2018.modules import SanakoyeuEtAl2018Discriminator
from pystiche.ops.comparison import MSEEncodingOperator
from pystiche.enc import Encoder
from pystiche.loss.perceptual import PerceptualLoss
from .utils import ContentOperatorContainer


def loss(
    predictions: List[torch.Tensor],
    real: bool,
    scale_weight: Optional[List[float]] = None,
) -> torch.Tensor:
    if scale_weight is None:
        scale_weight = [1.0] * len(predictions)
    assert len(scale_weight) == len(predictions)
    return torch.sum(
        torch.stack(
            [
                binary_cross_entropy_with_logits(
                    pred,
                    torch.ones_like(pred) if real else torch.zeros_like(pred),
                    reduction="sum",
                )
                * weight
                for pred, weight in zip(predictions, scale_weight)
            ]
        )
    )


def acc(predictions: List[torch.Tensor], real: bool) -> torch.Tensor:
    def get_acc_mask(pred: torch.Tensor, real: bool = True):
        if real:
            return torch.masked_fill(
                torch.zeros_like(pred), pred > torch.zeros_like(pred), 1
            )
        else:
            return torch.masked_fill(
                torch.zeros_like(pred), pred < torch.zeros_like(pred), 1
            )

    scale_acc = torch.sum(
        torch.stack([torch.mean(get_acc_mask(pred, real=real)) for pred in predictions])
    )
    return scale_acc / len(predictions)


class DiscriminatorOperator(RegularizationOperator):
    def __init__(
        self, discriminator: SanakoyeuEtAl2018Discriminator, score_weight: float = 1e0
    ) -> None:
        super().__init__(score_weight=score_weight)
        self.discriminator = discriminator
        self.acc = 0.0

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        input_pred = self.discriminator(image)
        return self.calculate_score(input_pred)

    def calculate_score(self, input_pred: List[torch.Tensor]) -> torch.Tensor:
        self.acc = acc(input_pred, real=True)
        return loss(input_pred, real=True)

    def get_current_acc(self):
        return self.acc


def sanakoyeu_et_al_2018_discriminator_loss(
    discriminator: SanakoyeuEtAl2018Discriminator,
    impl_params: bool = True,
    score_weight=None,
) -> RegularizationOperator:
    if score_weight is None:
        if impl_params:
            score_weight = 1e0
        else:
            score_weight = 1e-3
    return DiscriminatorOperator(discriminator, score_weight=score_weight)


def discriminator_loss(
    fake_image: torch.Tensor,
    real_image: torch.Tensor,
    discriminator: SanakoyeuEtAl2018Discriminator,
    scale_weight: Optional[List[float]] = None,
    fake_loss_correction_factor: float = 1.0,
    real_loss_correction_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    fake_predictions = discriminator(fake_image)
    real_predictions = discriminator(real_image)
    discr_loss = (
        loss(fake_predictions, real=False, scale_weight=scale_weight)
        * fake_loss_correction_factor
        + loss(real_predictions, real=True, scale_weight=scale_weight)
        * real_loss_correction_factor
    )
    discr_acc = (
        acc(fake_predictions, real=False) + acc(real_predictions, real=True)
    ) / 2
    return discr_loss, discr_acc


class DiscriminatorLoss(nn.Module):
    def __init__(
        self,
        discriminator: SanakoyeuEtAl2018Discriminator,
        fake_loss_correction_factor: float = 1.0,
        real_loss_correction_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.acc = 0.0

        self.fake_loss_correction_factor = fake_loss_correction_factor
        self.real_loss_correction_factor = real_loss_correction_factor

    def get_current_acc(self):
        return self.acc

    def forward(
        self, fake_image: torch.Tensor, real_image: torch.Tensor
    ) -> torch.Tensor:
        loss, self.acc = discriminator_loss(
            fake_image,
            real_image,
            self.discriminator,
            fake_loss_correction_factor=self.fake_loss_correction_factor,
            real_loss_correction_factor=self.real_loss_correction_factor,
        )
        return loss


class SanakoyeuEtAl2018FeatureOperator(MSEEncodingOperator):
    def __init__(
        self, encoder: Encoder, score_weight: float = 1.0, impl_params: bool = True
    ):
        super().__init__(encoder, score_weight=score_weight)
        self.impl_params = impl_params

    def calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        if self.impl_params:
            return torch.mean(torch.abs(input_repr - target_repr))
        else:
            return super().calculate_score(input_repr, target_repr, ctx)


def sanakoyeu_et_al_2018_style_aware_content_loss(
    encoder: Optional[pystiche.SequentialModule],
    impl_params: bool = True,
    score_weight=None,
) -> SanakoyeuEtAl2018FeatureOperator:
    if score_weight is None:
        if impl_params:
            score_weight = 1e2
        else:
            score_weight = 1e0

    return SanakoyeuEtAl2018FeatureOperator(
        encoder, score_weight=score_weight, impl_params=impl_params
    )


def sanakoyeu_et_al_2018_transformed_image_loss(
    transformer_block: Optional[SanakoyeuEtAl2018TransformerBlock] = None,
    impl_params: bool = True,
    score_weight=None,
) -> MSEEncodingOperator:
    if score_weight is None:
        if impl_params:
            score_weight = 1e2
        else:
            score_weight = 1e0

    if transformer_block is None:
        transformer_block = SanakoyeuEtAl2018TransformerBlock()

    return MSEEncodingOperator(transformer_block, score_weight=score_weight)


def sanakoyeu_et_al_2018_transformer_loss(
    encoder: Optional[pystiche.SequentialModule],
    discriminator: SanakoyeuEtAl2018Discriminator,
    impl_params: bool = True,
    style_aware_content_loss: Optional[SanakoyeuEtAl2018FeatureOperator] = None,
    transformed_image_loss: Optional[MSEEncodingOperator] = None,
    style_loss: Optional[DiscriminatorOperator] = None,
) -> PerceptualLoss:

    if style_aware_content_loss is None:
        style_aware_content_loss = sanakoyeu_et_al_2018_style_aware_content_loss(
            encoder, impl_params=impl_params
        )

    if transformed_image_loss is None:
        transformed_image_loss = sanakoyeu_et_al_2018_transformed_image_loss(
            impl_params=impl_params
        )

    content_loss = ContentOperatorContainer(
        OrderedDict(
            (
                ("style_aware_content_loss", style_aware_content_loss),
                ("tranformed_image_loss", transformed_image_loss),
            )
        )
    )

    if style_loss is None:
        style_loss = sanakoyeu_et_al_2018_discriminator_loss(
            discriminator, impl_params=impl_params
        )

    return PerceptualLoss(content_loss, style_loss)
