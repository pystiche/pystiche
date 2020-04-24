import itertools
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pystiche.loss.perceptual import PerceptualLoss
from pystiche.ops.container import MultiLayerEncodingOperator

from .data import (
    sanakoyeu_et_al_2018_dataset,
    sanakoyeu_et_al_2018_image_loader,
    sanakoyeu_et_al_2018_images,
)
from .loss import (
    MultiLayerDicriminatorEncodingOperator,
    SanakoyeuEtAl2018DiscriminatorLoss,
    sanakoyeu_et_al_2018_discriminator_operator,
    sanakoyeu_et_al_2018_transformer_loss,
)
from .modules import SanakoyeuEtAl2018Discriminator, SanakoyeuEtAl2018Transformer
from .utils import (
    ExponentialMovingAverage,
    sanakoyeu_et_al_2018_lr_scheduler,
    sanakoyeu_et_al_2018_optimizer,
)

__all__ = [
    "sanakoyeu_et_al_2018_training",
    "sanakoyeu_et_al_2018_stylization",
    "SanakoyeuEtAl2018Transformer",
    "SanakoyeuEtAl2018Discriminator",
    "sanakoyeu_et_al_2018_transformer_loss",
    "SanakoyeuEtAl2018DiscriminatorLoss",
    "sanakoyeu_et_al_2018_discriminator_operator",
    "sanakoyeu_et_al_2018_images",
    "sanakoyeu_et_al_2018_image_loader",
    "sanakoyeu_et_al_2018_dataset",
]


def gan_optim_loop(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    transformer: nn.Module,
    discriminator_operator: MultiLayerEncodingOperator,
    discriminator_criterion: nn.Module,
    transformer_criterion: nn.Module,
    transformer_criterion_update_fn: Callable[
        [nn.Module, nn.Module, torch.tensor, nn.ModuleDict], None
    ],
    discriminator_optimizer: Optional[Optimizer] = None,
    transformer_optimizer: Optional[Optimizer] = None,
    get_optimizer: Optional[Callable[[nn.Module], Optimizer]] = None,
    target_win_rate: float = 0.8,
    discriminator_success: Optional[ExponentialMovingAverage] = None,
    impl_params: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    if isinstance(style_image_loader, DataLoader):
        style_image_loader = itertools.cycle(style_image_loader)

    if discriminator_optimizer is None:
        discriminator_optimizer = get_optimizer(
            discriminator_operator.get_discriminator_parameters()
        )

    if transformer_optimizer is None:
        transformer_optimizer = get_optimizer(transformer)

    if discriminator_success is None:
        discriminator_success = ExponentialMovingAverage("discriminator_success")

    def train_discriminator_one_step(
        output_image: torch.Tensor,
        style_image: torch.Tensor,
        content_image: Optional[torch.Tensor] = None,
    ):
        def closure():
            discriminator_optimizer.zero_grad()
            loss = discriminator_criterion(output_image, style_image, content_image)
            loss.backward()
            return loss

        discriminator_optimizer.step(closure)
        discriminator_success.update(discriminator_criterion.get_current_acc())

    def train_transformer_one_step(output_image: torch.Tensor):
        def closure():
            transformer_optimizer.zero_grad()
            loss = transformer_criterion(output_image)
            loss.backward()
            return loss

        transformer_optimizer.step(closure)
        discriminator_success.update(
            (1.0 - transformer_criterion["style_loss"].get_discriminator_acc())
        )

    for content_image in content_image_loader:
        content_image = content_image.to(device)
        output_image = transformer(content_image)

        if discriminator_success.local_avg() < target_win_rate:
            style_image = next(style_image_loader)
            style_image = style_image.to(device)
            if impl_params:
                train_discriminator_one_step(
                    output_image, style_image, content_image=content_image
                )
            else:
                train_discriminator_one_step(output_image, style_image)
        else:
            transformer_criterion_update_fn(content_image, transformer_criterion)
            train_transformer_one_step(output_image)

    return transformer


def epoch_gan_optim_loop(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    transformer: nn.Module,
    discriminator_operator: MultiLayerEncodingOperator,
    epochs: int,
    discriminator_criterion: nn.Module,
    transformer_criterion: nn.Module,
    transformer_criterion_update_fn: Callable[
        [nn.Module, nn.Module, torch.tensor, nn.ModuleDict], None
    ],
    discriminator_optimizer: Optional[Optimizer] = None,
    transformer_optimizer: Optional[Optimizer] = None,
    discriminator_lr_scheduler: Optional[LRScheduler] = None,
    transformer_lr_scheduler: Optional[LRScheduler] = None,
    target_win_rate: float = 0.8,
    discriminator_success: Optional[ExponentialMovingAverage] = None,
    impl_params: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:

    style_image_loader = itertools.cycle(style_image_loader)

    if discriminator_success is None:
        discriminator_success = ExponentialMovingAverage("discriminator_success")

    if discriminator_optimizer is None:
        if discriminator_lr_scheduler is None:
            discriminator_optimizer = sanakoyeu_et_al_2018_optimizer(
                discriminator_operator.get_discriminator_parameters()
            )
        else:
            discriminator_optimizer = discriminator_lr_scheduler.optimizer

    if transformer_optimizer is None:
        if transformer_lr_scheduler is None:
            transformer_optimizer = sanakoyeu_et_al_2018_optimizer(transformer)
        else:
            transformer_optimizer = transformer_lr_scheduler.optimizer

    def optim_loop(transformer: nn.Module) -> nn.Module:
        return gan_optim_loop(
            content_image_loader,
            style_image_loader,
            transformer,
            discriminator_operator,
            discriminator_criterion,
            transformer_criterion,
            transformer_criterion_update_fn,
            discriminator_optimizer=discriminator_optimizer,
            transformer_optimizer=transformer_optimizer,
            target_win_rate=target_win_rate,
            discriminator_success=discriminator_success,
            impl_params=impl_params,
            device=device,
        )

    for epoch in range(epochs):
        transformer = optim_loop(transformer)

        if discriminator_lr_scheduler is not None:
            discriminator_lr_scheduler.step(epoch)

        if transformer_lr_scheduler is not None:
            transformer_lr_scheduler.step(epoch)

    return transformer


def sanakoyeu_et_al_2018_training(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    impl_params: bool = True,
    device: Optional[torch.device] = None,
    transformer: Optional[SanakoyeuEtAl2018Transformer] = None,
    discriminator_operator: Optional[MultiLayerDicriminatorEncodingOperator] = None,
    discriminator_criterion: Optional[SanakoyeuEtAl2018DiscriminatorLoss] = None,
    transformer_criterion: Optional[PerceptualLoss] = None,
    discriminator_lr_scheduler: Optional[ExponentialLR] = None,
    transformer_lr_scheduler: Optional[ExponentialLR] = None,
    num_epochs: Optional[int] = None,
    get_optimizer: Optional[sanakoyeu_et_al_2018_optimizer] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if transformer is None:
        transformer = SanakoyeuEtAl2018Transformer()
        transformer = transformer.train()
    transformer = transformer.to(device)

    if discriminator_operator is None:
        discriminator_operator = sanakoyeu_et_al_2018_discriminator_operator()

    if discriminator_criterion is None:
        discriminator_criterion = SanakoyeuEtAl2018DiscriminatorLoss(
            discriminator_operator
        )
        discriminator_criterion = discriminator_criterion.eval()
    discriminator_criterion = discriminator_criterion.to(device)

    if transformer_criterion is None:
        transformer_criterion = sanakoyeu_et_al_2018_transformer_loss(
            transformer.encoder,
            impl_params=impl_params,
            style_loss=discriminator_operator,
        )
        transformer_criterion = transformer_criterion.eval()
    transformer_criterion = transformer_criterion.to(device)

    if get_optimizer is None:
        get_optimizer = sanakoyeu_et_al_2018_optimizer

    def transformer_criterion_update_fn(content_image, criterion):
        criterion.set_content_image(content_image)

    if impl_params:
        return gan_optim_loop(
            content_image_loader,
            style_image_loader,
            transformer,
            discriminator_operator,
            discriminator_criterion,
            transformer_criterion,
            transformer_criterion_update_fn,
            get_optimizer=get_optimizer,
            impl_params=impl_params,
            device=device,
        )
    else:
        if discriminator_lr_scheduler is None:
            discriminator_optimizer = get_optimizer(
                discriminator_operator.get_discriminator_parameters()
            )
            discriminator_lr_scheduler = sanakoyeu_et_al_2018_lr_scheduler(
                discriminator_optimizer
            )

        if transformer_lr_scheduler is None:
            transformer_optimizer = get_optimizer(transformer)
            transformer_lr_scheduler = sanakoyeu_et_al_2018_lr_scheduler(
                transformer_optimizer
            )

        if num_epochs is None:
            num_epochs = 3

        return epoch_gan_optim_loop(
            content_image_loader,
            style_image_loader,
            transformer,
            discriminator_operator,
            num_epochs,
            discriminator_criterion,
            transformer_criterion,
            transformer_criterion_update_fn,
            discriminator_lr_scheduler=discriminator_lr_scheduler,
            transformer_lr_scheduler=transformer_lr_scheduler,
            impl_params=impl_params,
            device=device,
        )


def sanakoyeu_et_al_2018_stylization(
    input_image: torch.Tensor,
    transformer: Union[nn.Module, str],
    impl_params: bool = True,
):
    device = input_image.device
    if isinstance(transformer, str):
        transformer = SanakoyeuEtAl2018Transformer()
        if not impl_params:
            transformer = transformer.eval()
        transformer = transformer.to(device)

    with torch.no_grad():
        output_image = transformer(input_image)

    return output_image.detach()
