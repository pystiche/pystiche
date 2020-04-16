from typing import Optional, Callable, Union
import itertools
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.lr_scheduler import ExponentialLR
from pystiche.loss.perceptual import PerceptualLoss
from .modules import (
    SanakoyeuEtAl2018Generator,
    SanakoyeuEtAl2018Discriminator,
    sanakoyeu_et_al_2018_discriminator,
    SanakoyeuEtAl2018TransformerBlock,
)
from .loss import (
    DiscriminatorLoss,
    sanakoyeu_et_al_2018_generator_loss,
)
from .data import (
    sanakoyeu_et_al_2018_dataset,
    sanakoyeu_et_al_2018_image_loader,
    sanakoyeu_et_al_2018_images,
    sanakoyeu_et_al_2018_batch_sampler,
)
from .utils import sanakoyeu_et_al_2018_optimizer, ExponentialMovingAverage

__all__ = [
    "sanakoyeu_et_al_2018_training",
    "sanakoyeu_et_al_2018_stylization",
    "SanakoyeuEtAl2018Generator",
    "sanakoyeu_et_al_2018_discriminator",
    "SanakoyeuEtAl2018TransformerBlock",
    "sanakoyeu_et_al_2018_generator_loss",
    "DiscriminatorLoss",
    "sanakoyeu_et_al_2018_images",
    "sanakoyeu_et_al_2018_image_loader",
    "sanakoyeu_et_al_2018_dataset",
]


def gan_optim_loop(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    discriminator_criterion: nn.Module,
    generator_criterion: nn.Module,
    generator_criterion_update_fn: Callable[
        [nn.Module, nn.Module, torch.tensor, nn.ModuleDict], None
    ],
    discriminator_optimizer: Optional[Optimizer] = None,
    generator_optimizer: Optional[Optimizer] = None,
    get_optimizer: Optional[Callable[[nn.Module], Optimizer]] = None,
    target_win_rate: float = 0.8,
    discriminator_success: Optional[ExponentialMovingAverage] = None,
    impl_params: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:

    if generator_criterion is None:
        discriminator_optimizer = get_optimizer(discriminator)

    if generator_criterion is None:
        generator_criterion = get_optimizer(generator)

    if discriminator_success is None:
        discriminator_success = ExponentialMovingAverage("discriminator_success")

    def train_discriminator_one_step(fake_image, real_image):
        def closure():
            discriminator_optimizer.zero_grad()
            loss = discriminator_criterion(fake_image, real_image)
            loss.backward()
            return loss

        discriminator_optimizer.step(closure)
        discriminator_success.update(discriminator_criterion.get_current_acc())

    def train_generator_one_step(input_image):
        def closure():
            generator_optimizer.zero_grad()
            loss = generator_criterion(input_image)
            loss.backward()
            return loss

        generator_optimizer.step(closure)
        discriminator_success.update(
            (1.0 - generator_criterion["discr_loss"].get_current_acc())
        )

    for content_image in content_image_loader:
        content_image = content_image.to(device)
        stylized_image = generator(content_image)

        if discriminator_success.local_avg() < target_win_rate:
            style_image = next(style_image_loader)
            style_image = style_image.to(device)
            if impl_params:
                fake_images = torch.cat((stylized_image, content_image), 0)
            else:
                fake_images = stylized_image
            train_discriminator_one_step(fake_images, style_image)
        else:
            generator_criterion_update_fn(content_image, generator_criterion)
            train_generator_one_step(stylized_image)

    return generator


def epoch_gan_optim_loop(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    epochs: int,
    discriminator_criterion: nn.Module,
    generator_criterion: nn.Module,
    generator_criterion_update_fn: Callable[
        [nn.Module, nn.Module, torch.tensor, nn.ModuleDict], None
    ],
    discriminator_optimizer: Optional[Optimizer] = None,
    generator_optimizer: Optional[Optimizer] = None,
    discriminator_lr_scheduler: Optional[LRScheduler] = None,
    generator_lr_scheduler: Optional[LRScheduler] = None,
    target_win_rate: float = 0.8,
    impl_params: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:

    style_image_loader = itertools.cycle(style_image_loader)
    discriminator_success = ExponentialMovingAverage("discriminator_success")

    if discriminator_optimizer is None:
        if discriminator_lr_scheduler is None:
            discriminator_optimizer = sanakoyeu_et_al_2018_optimizer(discriminator)
        else:
            discriminator_optimizer = discriminator_lr_scheduler.optimizer

    if generator_optimizer is None:
        if generator_lr_scheduler is None:
            generator_optimizer = sanakoyeu_et_al_2018_optimizer(generator)
        else:
            generator_optimizer = generator_lr_scheduler.optimizer

    def update_batch_sampler(image_loader: DataLoader, num_batches: int = int(1e5)):
        batch_sampler = sanakoyeu_et_al_2018_batch_sampler(
            image_loader.dataset, num_batches=num_batches
        )
        return DataLoader(
            image_loader.dataset,
            batch_sampler=batch_sampler,
            num_workers=image_loader.num_workers,
            pin_memory=image_loader.pin_memory,
        )

    def optim_loop(generator: nn.Module) -> nn.Module:
        return gan_optim_loop(
            content_image_loader,
            style_image_loader,
            generator,
            discriminator,
            discriminator_criterion,
            generator_criterion,
            generator_criterion_update_fn,
            discriminator_optimizer=discriminator_optimizer,
            generator_optimizer=generator_optimizer,
            target_win_rate=target_win_rate,
            discriminator_success=discriminator_success,
            impl_params=impl_params,
            device=device,
        )

    for epoch in range(epochs):
        generator = optim_loop(generator)

        if discriminator_lr_scheduler is not None:
            discriminator_lr_scheduler.step(epoch)

        if generator_lr_scheduler is not None:
            generator_lr_scheduler.step(epoch)
        content_image_loader = update_batch_sampler(content_image_loader)

    return generator


def sanakoyeu_et_al_2018_training(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    impl_params: bool = True,
    device: Optional[torch.device] = None,
    generator: Optional[SanakoyeuEtAl2018Generator] = None,
    discriminator: Optional[SanakoyeuEtAl2018Discriminator] = None,
    transformer_block: Optional[SanakoyeuEtAl2018TransformerBlock] = None,
    discriminator_criterion: Optional[DiscriminatorLoss] = None,
    generator_criterion: Optional[PerceptualLoss] = None,
    discriminator_lr_scheduler: Optional[ExponentialLR] = None,
    generator_lr_scheduler: Optional[ExponentialLR] = None,
    num_epochs: Optional[int] = None,
    get_optimizer: Optional[sanakoyeu_et_al_2018_optimizer] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if generator is None:
        generator = SanakoyeuEtAl2018Generator()
        generator = generator.train()
    generator = generator.to(device)

    if discriminator is None:
        discriminator = sanakoyeu_et_al_2018_discriminator()
        discriminator = discriminator.train()
    discriminator = discriminator.to(device)

    if transformer_block is None:
        transformer_block = SanakoyeuEtAl2018TransformerBlock()
        if not impl_params:
            transformer_block = transformer_block.train()
    transformer_block = transformer_block.to(device)

    if discriminator_criterion is None:
        fake_loss_correction_factor = 2.0 if impl_params else 1.0
        discriminator_criterion = DiscriminatorLoss(discriminator, fake_loss_correction_factor=fake_loss_correction_factor)
        discriminator_criterion = discriminator_criterion.eval()
    discriminator_criterion = discriminator_criterion.to(device)

    if generator_criterion is None:
        generator_criterion = sanakoyeu_et_al_2018_generator_loss(
            generator.encoder, discriminator, transformer_block, impl_params=impl_params
        )
        generator_criterion = generator_criterion.eval()
    generator_criterion = generator_criterion.to(device)

    if get_optimizer is None:
        get_optimizer = sanakoyeu_et_al_2018_optimizer

    if discriminator_lr_scheduler is None:
        discriminator_optimizer = get_optimizer(discriminator)
        discriminator_lr_scheduler = ExponentialLR(discriminator_optimizer, 0.1)

    if generator_lr_scheduler is None:
        generator_optimizer = get_optimizer(generator)
        generator_lr_scheduler = ExponentialLR(generator_optimizer, 0.1)

    if num_epochs is None:
        if impl_params:
            num_epochs = 1
        else:
            num_epochs = 2

    def generator_criterion_update_fn(content_image, criterion):
        criterion.set_content_image(content_image)

    return epoch_gan_optim_loop(
        content_image_loader,
        style_image_loader,
        generator,
        discriminator,
        num_epochs,
        discriminator_criterion,
        generator_criterion,
        generator_criterion_update_fn,
        discriminator_lr_scheduler=discriminator_lr_scheduler,
        generator_lr_scheduler=generator_lr_scheduler,
        impl_params=impl_params,
        device=device,
    )


def sanakoyeu_et_al_2018_stylization(
    input_image: torch.Tensor,
    generator: Union[nn.Module, str],
    impl_params: bool = True,
):
    device = input_image.device
    if isinstance(generator, str):
        generator = SanakoyeuEtAl2018Generator()
        if not impl_params:
            generator = generator.eval()
        generator = generator.to(device)

    with torch.no_grad():
        output_image = generator(input_image)

    return output_image.detach()
