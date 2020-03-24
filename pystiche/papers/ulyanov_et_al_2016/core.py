from typing import Union, Optional, Callable, Tuple
import torch
from pystiche.image import CaffePostprocessing
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
import pystiche
from torch.optim.lr_scheduler import ExponentialLR
from pystiche.optim import (
    OptimLogger,
    default_transformer_epoch_optim_loop,
)
from ..common_utils import batch_up_image
from .modules import ulyanov_et_al_2016_transformer, UlyanovEtAl2016Transformer
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
    ulyanov_et_al_2016_lr_scheduler,
)


__all__ = [
    "ulyanov_et_al_2016_transformer",
    "ulyanov_et_al_2016_perceptual_loss",
    "ulyanov_et_al_2016_dataset",
    "ulyanov_et_al_2016_image_loader",
    "ulyanov_et_al_2016_training",
    "ulyanov_et_al_2016_images",
    "ulyanov_et_al_2016_style_generation",
    "ulyanov_et_al_2016_texture_generation",
]


def ulyanov_et_al_2016_training(
    content_image_loader: DataLoader,
    style: Union[str, torch.Tensor],
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    transformer: Optional[UlyanovEtAl2016Transformer] = None,
    criterion: Optional[UlyanovEtAl2016PerceptualLoss] = None,
    lr_scheduler: Optional[ExponentialLR] = None,
    num_epochs: Optional[int] = None,
    get_optimizer: Optional[Callable[[UlyanovEtAl2016Transformer], Optimizer]] = None,
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

    if lr_scheduler is None:
        if get_optimizer is None:
            get_optimizer = ulyanov_et_al_2016_optimizer
        optimizer = get_optimizer(transformer)

        lr_scheduler = ulyanov_et_al_2016_lr_scheduler(
            optimizer, impl_params=impl_params,
        )

    if num_epochs is None:
        if impl_params:
            if instance_norm:
                num_epochs = 25
            else:
                num_epochs = 10 if stylization else 5
        else:
            num_epochs = 10

    style_transform = ulyanov_et_al_2016_style_transform(
        impl_params=impl_params, instance_norm=instance_norm
    )
    style_transform = style_transform.to(device)
    preprocessor = ulyanov_et_al_2016_preprocessor()
    preprocessor = preprocessor.to(device)
    style_image = style_transform(style_image)
    style_image = preprocessor(style_image)
    style_image = batch_up_image(style_image, loader=content_image_loader)
    criterion.set_style_image(style_image)

    def criterion_update_fn(input_image, criterion, preprocessor=None):
        if hasattr(criterion, "content_loss"):
            if preprocessor is None:
                preprocessor = ulyanov_et_al_2016_preprocessor()
            preprocessor = preprocessor.to(device)
            criterion.set_content_image(preprocessor(input_image))

    default_transformer_epoch_optim_loop(
        content_image_loader,
        transformer,
        criterion,
        criterion_update_fn,
        num_epochs,
        device=device,
        lr_scheduler=lr_scheduler,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )

    return transformer


def ulyanov_et_al_2016_style_generation(
    input_image: torch.Tensor,
    transformer: Union[nn.Module, str],
    impl_params: bool = True,
    instance_norm: bool = False,
    stylization: bool = True,
    postprocessor: Optional[CaffePostprocessing] = None,
):
    device = input_image.device
    if isinstance(transformer, str):
        style = transformer
        transformer = ulyanov_et_al_2016_transformer(
            style=style, impl_params=impl_params, instance_norm=instance_norm, stylization=stylization
        )
        if instance_norm or not impl_params:
            transformer = transformer.eval()
        transformer = transformer.to(device)

    with torch.no_grad():
        content_transform = ulyanov_et_al_2016_content_transform(impl_params=impl_params,
                                                                 instance_norm=instance_norm)
        content_transform = content_transform.to(device)
        input_image = content_transform(input_image)

        if postprocessor is None:
            postprocessor = ulyanov_et_al_2016_postprocessor()
        postprocessor = postprocessor.to(device)
        output_image = transformer(input_image)
        output_image = torch.clamp(postprocessor(output_image), 0, 1)

    return output_image.detach()


def ulyanov_et_al_2016_texture_generation(
    input: Union[Tuple[int, int], torch.Tensor],
    transformer: Union[nn.Module, str],
    impl_params: bool = True,
    instance_norm: bool = False,
    stylization: bool = True,
    postprocessor: Optional[CaffePostprocessing] = None,
):
    if isinstance(input, torch.Tensor):
        device = input.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(transformer, str):
        style = transformer
        transformer = ulyanov_et_al_2016_transformer(
            style=style, impl_params=impl_params, instance_norm=instance_norm, stylization=stylization
        )
        if instance_norm or not impl_params:
            transformer = transformer.eval()
        transformer = transformer.to(device)

    with torch.no_grad():
        if postprocessor is None:
            postprocessor = ulyanov_et_al_2016_postprocessor()
        postprocessor = postprocessor.to(device)
        output_image = transformer(input)
        output_image = torch.clamp(postprocessor(output_image), 0, 1)

    return output_image.detach()
