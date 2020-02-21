from typing import Union, Optional, Iterable, Tuple, Callable
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import pystiche
from pystiche.image import extract_aspect_ratio, extract_image_size
from pystiche.pyramid import ImagePyramid
from pystiche.pyramid.level import PyramidLevel
from .log import (
    OptimLogger,
    default_image_optim_log_fn,
    default_pyramid_level_header,
    default_transformer_optim_log_fn,
)


def default_image_optimizer(input_image: torch.Tensor) -> Optimizer:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


def default_image_optim_loop(
    input_image: torch.Tensor,
    criterion: nn.Module,
    get_optimizer: Optional[Callable[[torch.Tensor], Optimizer]] = None,
    num_steps: Union[int, Iterable[int]] = 500,
    preprocessor: nn.Module = None,
    postprocessor: nn.Module = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict]], None]
    ] = None,
) -> torch.Tensor:
    if get_optimizer is None:
        get_optimizer = default_image_optimizer

    if isinstance(num_steps, int):
        num_steps = range(1, num_steps + 1)

    if logger is None:
        logger = OptimLogger()

    if log_fn is None:
        log_fn = default_image_optim_log_fn(optim_logger=logger)

    if preprocessor:
        with torch.no_grad():
            input_image = preprocessor(input_image)

    optimizer = get_optimizer(input_image)

    for step in num_steps:

        def closure():
            optimizer.zero_grad()

            loss = criterion(input_image)
            loss.backward()

            if not quiet:
                log_fn(step, loss)

            return loss.item()

        optimizer.step(closure)

    if postprocessor:
        with torch.no_grad():
            input_image = postprocessor(input_image)

    return input_image.detach()


def default_image_pyramid_optim_loop(
    input_image: torch.Tensor,
    criterion: nn.Module,
    pyramid: ImagePyramid,
    get_optimizer: Optional[Callable[[torch.Tensor], Optimizer]] = None,
    preprocessor: nn.Module = None,
    postprocessor: nn.Module = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    get_pyramid_level_header: Optional[
        Callable[[int, PyramidLevel, Tuple[int, int]], str]
    ] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict]], None]
    ] = None,
) -> torch.Tensor:
    aspect_ratio = extract_aspect_ratio(input_image)
    if get_optimizer is None:
        get_optimizer = default_image_optimizer

    if logger is None:
        logger = OptimLogger()

    if get_pyramid_level_header is None:
        get_pyramid_level_header = default_pyramid_level_header

    output_image = input_image
    for num, level in enumerate(pyramid, 1):

        def image_optim_loop(input_image):
            return default_image_optim_loop(
                input_image,
                criterion,
                get_optimizer=get_optimizer,
                num_steps=iter(level),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                quiet=quiet,
                logger=logger,
                log_fn=log_fn,
            )

        with torch.no_grad():
            input_image = level.resize_image(output_image, aspect_ratio=aspect_ratio)

        if quiet:
            output_image = image_optim_loop(input_image)
        else:
            input_image_size = extract_image_size(input_image)
            header = get_pyramid_level_header(num, level, input_image_size)
            with logger.environment(header):
                output_image = image_optim_loop(input_image)

    return output_image


def default_transformer_optimizer(transformer: nn.Module) -> Optimizer:
    return optim.Adam(transformer.parameters(), lr=1e-3)


def default_transformer_optim_loop(
    image_loader: DataLoader,
    device: torch.device,
    transformer: nn.Module,
    criterion: nn.Module,
    criterion_update_fn: Callable[[torch.Tensor, nn.ModuleDict], None],
    get_optimizer: Optional[Callable[[nn.Module], Optimizer]] = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> nn.Module:
    if get_optimizer is None:
        get_optimizer = default_transformer_optimizer

    if logger is None:
        logger = OptimLogger()

    if log_fn is None:
        log_fn = default_transformer_optim_log_fn(logger, len(image_loader))

    optimizer = get_optimizer(transformer)

    loading_time_start = time.time()
    for batch, input_image in enumerate(image_loader, 1):
        input_image = input_image.to(device)

        criterion_update_fn(input_image, criterion)

        loading_time = time.time() - loading_time_start

        def closure():
            processing_time_start = time.time()

            optimizer.zero_grad()

            output_image = transformer(input_image)
            loss = criterion(output_image)
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

    return transformer
