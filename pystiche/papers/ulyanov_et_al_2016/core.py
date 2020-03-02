from typing import Union, Optional, Callable
import torch
from torch.utils.data import DataLoader
from torch import nn
import time
from torch.optim.optimizer import Optimizer
import pystiche
from pystiche.optim import (
    OptimLogger,
    default_transformer_optim_log_fn,
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
    ulyanov_et_al_2016_preprocessor_Criterion,
    ulyanov_et_al_2016_optimizer,
)

__all__ = [
    "ulyanov_et_al_2016_transformer_optim_loop",
    "ulyanov_et_al_2016_transformer",
    "ulyanov_et_al_2016_perceptual_loss",
    "ulyanov_et_al_2016_dataset",
    "ulyanov_et_al_2016_image_loader",
    "ulyanov_et_al_2016_training",
    "ulyanov_et_al_2016_images",
    "ulyanov_et_al_2016_stylization",
]


def ulyanov_et_al_2016_transformer_optim_loop(
    image_loader: DataLoader,
    device: torch.device,
    transformer: nn.Module,
    criterion: nn.Module,
    criterion_update_fn: Callable[[torch.Tensor, nn.ModuleDict], None],
    get_optimizer: ulyanov_et_al_2016_optimizer,
    preprocess_Criterion: ulyanov_et_al_2016_preprocessor_Criterion = None,
    impl_params: bool = True,
    mode: str = "texture",
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> nn.Module:

    if logger is None:
        logger = OptimLogger()

    if log_fn is None:
        log_fn = default_transformer_optim_log_fn(logger, len(image_loader))

    optimizer = get_optimizer(transformer, impl_params=impl_params)

    loading_time_start = time.time()
    for batch, input_image in enumerate(image_loader, 1):
        input_image = input_image.to(device)

        if mode == "style":
            criterion_update_fn(input_image, criterion)

        loading_time = time.time() - loading_time_start

        def closure():
            processing_time_start = time.time()

            optimizer.zero_grad()

            output_image = transformer(input_image)
            if preprocess_Criterion is not None:
                output_image = preprocess_Criterion(output_image)
            loss = criterion(output_image)
            for key in loss.keys():
                loss[key] = loss[key] / output_image.size()[0]
            loss.backward()

            processing_time = time.time() - processing_time_start

            if not quiet:
                batch_size = input_image.size()[0]
                image_loading_velocity = batch_size / loading_time
                image_processing_velocity = batch_size / processing_time
                log_fn(batch, loss, image_loading_velocity, image_processing_velocity)

            return loss

        optimizer.step(closure)
        loading_time_start = time.time()

        # change learning rate during training
        if impl_params:
            if batch % 2000 == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.8
        else:
            if batch % 200 == 0 and batch >= 1000:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.7

    return transformer


def ulyanov_et_al_2016_training(
    content_image_loader: DataLoader,
    style: Union[str, torch.Tensor],
    impl_params=True,
    instance_norm: bool = True,
    mode: str = "texture",
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

    preprocessor_Criterion = ulyanov_et_al_2016_preprocessor_Criterion()
    preprocessor_Criterion = preprocessor_Criterion.to(device)
    style_image = preprocessor_Criterion(style_image)

    if transformer is None:
        transformer = ulyanov_et_al_2016_transformer(
            impl_params=impl_params,
            instance_norm=instance_norm,
            mode=mode,
            device=device,
        )
        transformer = transformer.train()
    transformer = transformer.to(device)

    if criterion is None:
        criterion = ulyanov_et_al_2016_perceptual_loss(
            impl_params=impl_params, mode=mode
        )
        criterion = criterion.eval()
    criterion = criterion.to(device)

    if get_optimizer is None:
        get_optimizer = ulyanov_et_al_2016_optimizer

    style_transform = ulyanov_et_al_2016_style_transform(impl_params=impl_params)
    style_transform = style_transform.to(device)
    style_image = style_transform(style_image)
    style_image = batch_up_image(style_image, loader=content_image_loader)

    criterion.set_style_image(style_image)

    def criterion_update_fn(input_image, criterion):
        criterion.set_content_image(input_image)

    ulyanov_et_al_2016_transformer_optim_loop(
        content_image_loader,
        device,
        transformer,
        criterion,
        criterion_update_fn,
        get_optimizer=get_optimizer,
        preprocess_Criterion=preprocessor_Criterion,
        impl_params=impl_params,
        mode=mode,
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
    weights: str = "pystiche",
):
    device = input_image.device

    if isinstance(transformer, str):
        style = transformer
        transformer = ulyanov_et_al_2016_transformer(
            style=style,
            weights=weights,
            impl_params=impl_params,
            instance_norm=instance_norm,
        )
        transformer = transformer.eval()
        transformer = transformer.to(device)

    with torch.no_grad():
        # transform to 256x256 -> paper same
        content_transform = ulyanov_et_al_2016_content_transform(
            impl_params=impl_params
        )
        content_transform = content_transform.to(device)
        input_image = content_transform(input_image)

        output_image = transformer(input_image)

    return output_image.detach()
