from typing import Optional, Callable, Union
import itertools
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
from .modules import (
    SanakoyeuEtAl2018Generator,
    SanakoyeuEtAl2018Discriminator,
    sanakoyeu_et_al_2018_discriminator,
    SanakoyeuEtAl2018TransformerBlock,
)
from .loss import (
    SanakoyeuEtAl2018GeneratorLoss,
    DiscriminatorLoss,
    sanakoyeu_et_al_2018_generator_loss,
)
from .data import (
    sanakoyeu_et_al_2018_dataset,
    sanakoyeu_et_al_2018_image_loader,
    sanakoyeu_et_al_2018_images,
)
from .utils import sanakoyeu_et_al_2018_optimizer

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
    "sanakoyeu_et_al_2018_dataset"
]


def default_gan_optim_loop(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    transformer_block: nn.Module,
    discriminator_criterion: nn.Module,
    generator_criterion: nn.Module,
    generator_criterion_update_fn: Callable[
        [nn.Module, nn.Module, torch.tensor, nn.ModuleDict], None
    ],
    init_discr_success: float = 0.8,
    init_win_rate: float = 0.8,
    alpha: float = 0.05,
    impl_params: bool = True,
    get_optimizer: Optional[Callable[[nn.Module], Optimizer]] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:

    discriminator_optimizer = get_optimizer(discriminator)
    generator_optimizer = get_optimizer(generator)
    style_image_loader = itertools.cycle(style_image_loader)

    discr_success = init_discr_success
    win_rate = init_win_rate

    def train_discriminator_one_step(fake_image, real_image, discr_success):
        def closure():
            discriminator_optimizer.zero_grad()
            loss = discriminator_criterion(fake_image, real_image)
            loss.backward()
            return loss

        discriminator_optimizer.step(closure)
        acc = discriminator_criterion.get_current_acc()
        discr_success = discr_success * (1.0 - alpha) + alpha * acc

        return discr_success

    def train_generator_one_step(input_image, discr_success):
        def closure():
            generator_optimizer.zero_grad()
            loss = generator_criterion(input_image)
            loss.backward()
            return loss

        generator_optimizer.step(closure)
        acc = generator_criterion["discr_loss"].get_current_acc()
        discr_success = discr_success * (1.0 - alpha) + alpha * (1.0 - acc)
        return discr_success

    for content_image in content_image_loader:
        content_image = content_image.to(device)
        stylized_image = generator(content_image)

        if discr_success < win_rate:
            style_image = next(style_image_loader)
            style_image = style_image.to(device)
            if impl_params:
                fake_images = torch.cat((stylized_image, content_image), 0)
            else:
                fake_images = stylized_image
            discr_success = train_discriminator_one_step(
                fake_images, style_image, discr_success
            )
        else:
            generator_criterion_update_fn(
                generator.encoder, transformer_block, content_image, generator_criterion
            )
            discr_success = train_generator_one_step(stylized_image, discr_success)

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
    generator_criterion: Optional[SanakoyeuEtAl2018GeneratorLoss] = None,
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
        discriminator_criterion = DiscriminatorLoss(discriminator)
        discriminator_criterion = discriminator_criterion.eval()
    discriminator_criterion = discriminator_criterion.to(device)

    if generator_criterion is None:
        generator_criterion = sanakoyeu_et_al_2018_generator_loss(
            generator.encoder, discriminator, impl_params=impl_params
        )
        generator_criterion = generator_criterion.eval()
    generator_criterion = generator_criterion.to(device)

    if get_optimizer is None:
        get_optimizer = sanakoyeu_et_al_2018_optimizer

    def generator_criterion_update_fn(
        encoder, transformer_block, content_image, criterion
    ):
        criterion.update_encoder(encoder)
        criterion.update_transformer(transformer_block)
        criterion.set_content_image(content_image)

    return default_gan_optim_loop(
        content_image_loader,
        style_image_loader,
        generator,
        discriminator,
        transformer_block,
        discriminator_criterion,
        generator_criterion,
        generator_criterion_update_fn,
        impl_params=impl_params,
        get_optimizer=get_optimizer,
        device=device,
    )

def sanakoyeu_et_al_2018_stylization(
    input_image: torch.Tensor,
    generator: Union[nn.Module, str],
    impl_params: bool = True
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
