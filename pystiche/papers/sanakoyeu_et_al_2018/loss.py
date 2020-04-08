from typing import List, Optional, Tuple
from pystiche.papers.sanakoyeu_et_al_2018.modules import (
    SanakoyeuEtAl2018TransformerBlock,
)
import pystiche.ops.functional as F
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from pystiche.ops import RegularizationOperator
from pystiche.papers.sanakoyeu_et_al_2018.modules import SanakoyeuEtAl2018Discriminator, SanakoyeuEtAl2018Encoder
from pystiche.ops.op import EncodingComparisonOperator
from pystiche.enc import Encoder


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
                    pred, torch.ones_like(pred) if real else torch.zeros_like(pred),
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
        input_repr = self.discriminator(image)
        return self.calculate_score(input_repr)

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        self.acc = acc([input_repr], real=True)
        return loss([input_repr], real=True)

    def get_current_acc(self):
        return self.acc

    def set_current_discriminator(self, discriminator: nn.Module):
        self.discriminator = discriminator  # FIXME: right?


def sanakoyeu_et_al_2018_discriminator_loss(
    impl_params: bool = True,
    discriminator: Optional[SanakoyeuEtAl2018Discriminator] = None,
    score_weight=None,
) -> RegularizationOperator:
    if score_weight is None:
        if impl_params:
            score_weight = 1e0
        else:
            score_weight = 1e-3

    if discriminator is None:
        discriminator = SanakoyeuEtAl2018Discriminator()

    return DiscriminatorOperator(discriminator, score_weight=score_weight)


def discriminator_loss(
    fake_image: torch.Tensor,
    real_image: torch.Tensor,
    discriminator: SanakoyeuEtAl2018Discriminator,
    scale_weight: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    fake_predictions = discriminator(fake_image)
    real_predictions = discriminator(real_image)
    discr_loss = loss(fake_predictions, real=False, scale_weight=scale_weight) + loss(
        real_predictions, real=True, scale_weight=scale_weight
    )
    discr_acc = (acc(fake_predictions, real=False) + acc(real_predictions, real=True)) / 2
    return discr_loss, discr_acc


class DiscriminatorLoss(nn.Module):
    def __init__(self, discriminator: SanakoyeuEtAl2018Discriminator) -> None:
        super().__init__()
        self.discriminator = discriminator

    def forward(
        self, fake_image: torch.Tensor, real_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return discriminator_loss(fake_image, real_image, self.discriminator)


class SanakoyeuEtAl2018FeatureOperator(EncodingComparisonOperator):
    def __init__(self, encoder: Encoder, score_weight: float = 1.0, impl_params: bool = True):
        super().__init__(encoder, score_weight=score_weight)
        self.impl_params = impl_params

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return enc

    def input_enc_to_repr(self, enc: torch.Tensor, ctx: None) -> torch.Tensor:
        return self.enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc), None

    def update_encoder(self, encoder: Encoder):
        self.encoder = encoder

    def calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        if self.impl_params:
            return torch.mean(torch.abs(input_repr - target_repr))
        else:
            return F.mse_loss(input_repr, target_repr)


def sanakoyeu_et_al_2018_style_aware_content_loss(
    impl_params: bool = True,
    encoder: Optional[SanakoyeuEtAl2018Encoder] = None,
    score_weight=None,
) -> SanakoyeuEtAl2018FeatureOperator:
    if score_weight is None:
        if impl_params:
            score_weight = 1e2
        else:
            score_weight = 1e0

    if encoder is None:
        encoder = SanakoyeuEtAl2018Encoder()

    return SanakoyeuEtAl2018FeatureOperator(encoder, score_weight=score_weight, impl_params=impl_params)
# --------------------------------------------------------------------------------------------------------------------


def sanakoyeu_et_al_2018_image_loss(
    input_image: torch.Tensor,
    output_image: torch.Tensor,
    transformer_block: SanakoyeuEtAl2018TransformerBlock,
) -> torch.Tensor:
    return F.mse_loss(transformer_block(input_image), transformer_block(output_image))
