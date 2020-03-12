from typing import Union, Optional, Callable
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler
import pystiche
from pystiche.optim import (
    OptimLogger,
    default_transformer_optim_loop,
)
from ..common_utils import batch_up_image
from .modules import ulyanov_et_al_2016_transformer
from .loss import (
    UlyanovEtAl2016PerceptualLoss,
    ulyanov_et_al_2016_perceptual_loss,
)

from .data import (
    ulyanov_et_al_2016_style_transform,
    ulyanov_et_al_2016_content_transform,
    ulyanov_et_al_2016_dataset,
    ulyanov_et_al_2016_image_loader,
    ulyanov_et_al_2016_images,
)
from .utils import (
    ulyanov_et_al_2016_preprocessor,
    ulyanov_et_al_2016_postprocessor,
    ulyanov_et_al_2016_optimizer,
)

__all__ = [
    "ulyanov_et_al_2016_transformer",
    "ulyanov_et_al_2016_perceptual_loss",
    "ulyanov_et_al_2016_dataset",
    "ulyanov_et_al_2016_image_loader",
    "ulyanov_et_al_2016_training",
    "ulyanov_et_al_2016_images",
    "ulyanov_et_al_2016_stylization",
]


def ulyanov_et_al_2016_transformer_epoch_optim_loop(
    image_loader: DataLoader,
    device: torch.device,
    transformer: nn.Module,
    criterion: nn.Module,
    criterion_update_fn: Callable[[torch.Tensor, nn.ModuleDict], None],
    get_optimizer: ulyanov_et_al_2016_optimizer,
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> None:

    if impl_params:
        if instance_norm:
            num_epoch = 25
        else:
            num_epoch = 10 if stylization else 5
        gamma = 0.8
    else:
        num_epoch = 10
        gamma = 0.7

    if get_optimizer is None:
        get_optimizer = ulyanov_et_al_2016_optimizer(
            transformer, impl_params=impl_params, instance_norm=instance_norm
        )
    optimizer = get_optimizer(transformer)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(num_epoch):
        default_transformer_optim_loop(
            image_loader,
            device,
            transformer,
            criterion,
            criterion_update_fn,
            get_optimizer=optimizer,
            quiet=quiet,
            logger=logger,
            log_fn=log_fn,
        )
        if not impl_params:
            if epoch >= 5:  # start to reduce lr after 1000 steps
                scheduler.step()  # TODO: check this
        else:
            scheduler.step()


def ulyanov_et_al_2016_training(
    content_image_loader: DataLoader,
    style: Union[str, torch.Tensor],
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    transformer: Optional[ulyanov_et_al_2016_transformer] = None,
    criterion: Optional[UlyanovEtAl2016PerceptualLoss] = None,
    get_optimizer: Optional[
        Callable[[ulyanov_et_al_2016_transformer], Optimizer]
    ] = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
):
    if isinstance(style, str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = ulyanov_et_al_2016_images(download=False)
        style_image = images[style].read(device=device)
    else:
        style_image = style
        device = style_image.device

    if transformer is None:
        transformer = ulyanov_et_al_2016_transformer(
            impl_params=impl_params,
            instance_norm=instance_norm,
            stylization=stylization,
        )
        transformer = transformer.train()
    transformer = transformer.to(device)

    if criterion is None:
        criterion = ulyanov_et_al_2016_perceptual_loss(
            impl_params=impl_params,
            instance_norm=instance_norm,
            stylization=stylization,
        )
        criterion = criterion.eval()
    criterion = criterion.to(device)

    if get_optimizer is None:
        get_optimizer = ulyanov_et_al_2016_optimizer

    style_transform = ulyanov_et_al_2016_style_transform(
        impl_params=impl_params, instance_norm=instance_norm
    )
    style_transform = style_transform.to(device)
    style_image = style_transform(style_image)
    style_image = batch_up_image(style_image, loader=content_image_loader)

    criterion.set_style_image(style_image)

    def criterion_update_fn(input_image, criterion, preprocessor=None):
        if hasattr(criterion, "content_loss"):
            if preprocessor is None:
                preprocessor = ulyanov_et_al_2016_preprocessor()
            preprocessor = preprocessor.to(device)
            criterion.set_content_image(preprocessor(input_image))

    ulyanov_et_al_2016_transformer_epoch_optim_loop(
        content_image_loader,
        device,
        transformer,
        criterion,
        criterion_update_fn,
        get_optimizer=get_optimizer,
        impl_params=impl_params,
        instance_norm=instance_norm,
        stylization=stylization,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )

    return transformer


def ulyanov_et_al_2016_stylization(
    input_image: torch.Tensor,
    transformer: Union[nn.Module, str],
    impl_params: bool = True,
    instance_norm: bool = None,
    sample_size: int = 256,
    postprocessor: ulyanov_et_al_2016_postprocessor = None,
):
    device = input_image.device
    if isinstance(transformer, str):
        style = transformer
        transformer = ulyanov_et_al_2016_transformer(
            style=style, impl_params=impl_params, instance_norm=instance_norm,
        )
        transformer = transformer.eval()
        transformer = transformer.to(device)

    with torch.no_grad():
        content_transform = ulyanov_et_al_2016_content_transform(
            edge_size=sample_size, impl_params=impl_params
        )
        if postprocessor is None:
            postprocessor = ulyanov_et_al_2016_postprocessor()
        content_transform = content_transform.to(device)
        postprocessor = postprocessor.to(device)
        input_image = content_transform(input_image)
        output_image = transformer(input_image)
        output_image = torch.clamp(postprocessor(output_image), 0, 1)

    return output_image.detach()
