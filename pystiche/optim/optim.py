import time
from typing import Callable, Iterable, Optional, Tuple, Union, cast

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pystiche
from pystiche.image import extract_aspect_ratio, extract_image_size
from pystiche.pyramid import ImagePyramid
from pystiche.pyramid.level import PyramidLevel

from .log import (
    OptimLogger,
    default_epoch_header,
    default_image_optim_log_fn,
    default_pyramid_level_header,
    default_transformer_optim_log_fn,
)


def default_image_optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    r"""
    Args:
        input_image: Image to be optimized.

    Returns:
        :class:`torch.optim.LBFGS` optimizer with a learning rate of ``1.0``. The
        pixels of ``input_image`` are set as optimization parameters.
    """
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


def default_image_optim_loop(
    input_image: torch.Tensor,
    criterion: nn.Module,
    get_optimizer: Optional[Callable[[torch.Tensor], Optimizer]] = None,
    num_steps: Union[int, Iterable[int]] = 500,
    preprocessor: Optional[nn.Module] = None,
    postprocessor: Optional[nn.Module] = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict]], None]
    ] = None,
) -> torch.Tensor:
    r"""Perform an image optimization with integrated logging.

    Args:
        input_image: Image to be optimized.
        criterion: Optimization criterion.
        get_optimizer: Optional getter for the optimizer. If ``None``,
            :func:`default_image_optimizer` is used. Defaults to ``None``.
        num_steps: Number of optimization steps. Defaults to ``500``.
        preprocessor: Optional preprocessor that is called with the ``input_image``
            before the optimization.
        postprocessor: Optional preprocessor that is called with the ``input_image``
            after the optimization.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If ``None``,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current step and loss. If ``None``,
            :func:`pystiche.optim.default_image_optim_log_fn` is used. Defaults to
            ``None``.
    """
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

        def closure() -> float:
            optimizer.zero_grad()

            loss = criterion(input_image)
            loss.backward()

            if not quiet:
                # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
                log_fn(step, loss)  # type: ignore[misc]

            return cast(float, loss.item())

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
    preprocessor: Optional[nn.Module] = None,
    postprocessor: Optional[nn.Module] = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    get_pyramid_level_header: Optional[
        Callable[[int, PyramidLevel, Tuple[int, int]], str]
    ] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict]], None]
    ] = None,
) -> torch.Tensor:
    r"""Perform a image optimization for :class:`pystiche.pyramid.ImagePyramid` s with
    integrated logging.

    Args:
        input_image: Image to be optimized.
        criterion: Optimization criterion.
        pyramid: Image pyramid.
        get_optimizer: Optional getter for the optimizer. If ``None``,
            :func:`default_image_optimizer` is used. Defaults to ``None``.
        preprocessor: Optional preprocessor that is called with the ``input_image``
            before the optimization.
        postprocessor: Optional preprocessor that is called with the ``input_image``
            after the optimization.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If ``None``,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        get_pyramid_level_header: Optional custom getter for the logged pyramid level
            header. It is called before each level with the current level number,
            the :class:`pystiche.pyramid.PyramidLevel`, and the size of the
            ``input_image``. If ``None``
            :func:`pystiche.optim.default_pyramid_level_header` is used. Defaults to
            ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current step and loss. If ``None``,
            :func:`pystiche.optim.default_image_optim_log_fn` is used. Defaults to
            ``None``.
    """
    aspect_ratio = extract_aspect_ratio(input_image)
    if get_optimizer is None:
        get_optimizer = default_image_optimizer

    if logger is None:
        logger = OptimLogger()

    if get_pyramid_level_header is None:
        get_pyramid_level_header = default_pyramid_level_header

    output_image = input_image
    for num, level in enumerate(pyramid, 1):

        def image_optim_loop(input_image: torch.Tensor) -> torch.Tensor:
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
    r"""
    Args:
        transformer: Transformer to be optimized.

    Returns:
        :class:`torch.optim.Adam` optimizer with a learning rate of ``1e-3``. The
        parameters of ``transformer`` are set as optimization parameters.
    """
    return optim.Adam(transformer.parameters(), lr=1e-3)


def default_transformer_optim_loop(
    image_loader: DataLoader,
    transformer: nn.Module,
    criterion: nn.Module,
    criterion_update_fn: Callable[[torch.Tensor, nn.Module], None],
    optimizer: Optional[Optimizer] = None,
    get_optimizer: Optional[Callable[[nn.Module], Optimizer]] = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> nn.Module:
    r"""Perform a transformer optimization for a single epoch with integrated logging.

    Args:
        image_loader: Images used as input for the ``transformer``. Drawing from this
            should yield a single item.
        transformer: Transformer to be optimized.
        criterion: Optimization criterion.
        criterion_update_fn: Is called before each optimization step with the current
            images and the optimization ``criterion``.
        optimizer: Optional optimizer. If ``None``,
            :func:`default_transformer_optimizer` is used.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If ``None``,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current batch, loss as well as the image loading and
            processing velocities in img/s. If ``None``,
            :func:`pystiche.optim.default_transformer_optim_log_fn` is used. Defaults
            to ``None``.
    """
    if optimizer is None:
        optimizer = default_transformer_optimizer(transformer)

    if logger is None:
        logger = OptimLogger()

    if log_fn is None:
        log_fn = default_transformer_optim_log_fn(logger, len(image_loader))

    device = next(transformer.parameters()).device

    loading_time_start = time.time()
    for batch, input_image in enumerate(image_loader, 1):
        input_image = input_image.to(device)

        criterion_update_fn(input_image, criterion)

        loading_time = time.time() - loading_time_start

        def closure() -> float:
            processing_time_start = time.time()

            # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
            optimizer.zero_grad()  # type: ignore[union-attr]

            output_image = transformer(input_image)
            loss = criterion(output_image)
            loss.backward()

            processing_time = time.time() - processing_time_start

            if not quiet:
                batch_size = input_image.size()[0]
                image_loading_velocity = batch_size / max(loading_time, 1e-6)
                image_processing_velocity = batch_size / max(processing_time, 1e-6)
                # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
                log_fn(batch, loss, image_loading_velocity, image_processing_velocity)  # type: ignore[misc]

            return cast(float, loss.item())

        optimizer.step(closure)
        loading_time_start = time.time()

    return transformer


def default_transformer_epoch_optim_loop(
    image_loader: DataLoader,
    transformer: nn.Module,
    criterion: nn.Module,
    criterion_update_fn: Callable[[torch.Tensor, nn.Module], None],
    epochs: int,
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[LRScheduler] = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    get_epoch_header: Optional[
        Callable[[int, Optimizer, Optional[LRScheduler]], str]
    ] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> nn.Module:
    r"""Perform a transformer optimization for multiple epochs with integrated logging.

    Args:
        image_loader: Images used as input for the ``transformer``. Drawing from this
            should yield a single item.
        transformer: Transformer to be optimized.
        criterion: Optimization criterion.
        criterion_update_fn: Is called before each optimization step with the current
            images and the optimization ``criterion``.
        epochs: Number of epochs.
        optimizer: Optional optimizer. If ``None``, it is extracted from
            ``lr_scheduler`` or func:`default_transformer_optimizer` is used.
        lr_scheduler: Optional learning rate scheduler. ``step()`` is invoked after
            every epoch.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If ``None``,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        get_epoch_header: Optional custom getter for the logged epoch header. It is
        called before each epoch with the current epoch number, the ``optimizer``, and
            the ``lr_scheduler``. If ``None``
            :func:`pystiche.optim.default_epoch_header` is used. Defaults to ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current batch, loss as well as the image loading and
            processing velocities in img/s. If ``None``,
            :func:`pystiche.optim.default_transformer_optim_log_fn` is used. Defaults
            to ``None``.
        """
    if optimizer is None:
        if lr_scheduler is None:
            optimizer = default_transformer_optimizer(transformer)
        else:
            # Every LRScheduler has optimizer as a valid attribute
            # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            # but this is not reflected in the torch type hints
            optimizer = lr_scheduler.optimizer  # type: ignore[attr-defined]

    if get_epoch_header is None:
        get_epoch_header = default_epoch_header

    def transformer_optim_loop(transformer: nn.Module) -> nn.Module:
        # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
        return default_transformer_optim_loop(
            image_loader,
            transformer,
            criterion,
            criterion_update_fn,
            optimizer,
            quiet=quiet,
            logger=logger,
            log_fn=log_fn,
        )

    for epoch in range(epochs):
        if quiet:
            transformer = transformer_optim_loop(transformer)
        else:
            header = get_epoch_header(epoch, optimizer, lr_scheduler)
            # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
            with logger.environment(header):  # type: ignore[union-attr]
                transformer = transformer_optim_loop(transformer)

        if lr_scheduler is not None:
            lr_scheduler.step()

    return transformer
