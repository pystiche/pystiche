import sys
import warnings
from typing import Any, Callable, Iterable, Iterator, Optional, Union

from tqdm.auto import tqdm

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pystiche
from pystiche import loss as loss_
from pystiche.image import extract_aspect_ratio
from pystiche.misc import build_deprecation_message
from pystiche.pyramid import ImagePyramid

__all__ = [
    "default_image_optimizer",
    "image_optimization",
    "pyramid_image_optimization",
    "default_model_optimizer",
    "model_optimization",
    "multi_epoch_model_optimization",
]


def default_image_optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    r"""
    Args:
        input_image: Image to be optimized.

    Returns:
        :class:`torch.optim.LBFGS` optimizer with a learning rate of ``1.0``. The
        pixels of ``input_image`` are set as optimization parameters.
    """
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


class OptimProgressBar(tqdm):
    def __init__(
        self,
        name: str,
        total_or_iterable: Union[int, Iterable],
        quiet: bool = False,
        **kwargs: Any,
    ) -> None:
        iterable = (
            range(total_or_iterable)
            if isinstance(total_or_iterable, int)
            else total_or_iterable
        )
        super().__init__(
            desc=name, iterable=iterable, disable=quiet, file=sys.stdout, **kwargs,
        )

    def update(
        self, n: int = 1, loss: Optional[Union[float, pystiche.LossDict]] = None
    ) -> None:
        if loss is not None:
            self.set_postfix(loss=f"{float(loss):.3e}", refresh=False)
        super().update(n)


def image_optimization(
    input_image: torch.Tensor,
    criterion: nn.Module,
    optimizer: Optional[Union[Optimizer, Callable[[torch.Tensor], Optimizer]]] = None,
    num_steps: Union[int, Iterable[int]] = 500,
    preprocessor: Optional[nn.Module] = None,
    postprocessor: Optional[nn.Module] = None,
    quiet: bool = False,
) -> torch.Tensor:
    r"""Perform an image optimization with integrated logging.

    Args:
        input_image: Image to be optimized.
        criterion: Optimization criterion.
        optimizer: Optional optimizer or optimizer getter. If omitted,
            :func:`default_image_optimizer` is used. If a ``preprocessor`` is used, has
            to be a getter.
        num_steps: Number of optimization steps. Defaults to ``500``.
        preprocessor: Optional preprocessor that is called with the ``input_image``
            before the optimization.
        postprocessor: Optional preprocessor that is called with the ``input_image``
            after the optimization.
        quiet: If ``True``, no information is printed to STDOUT during the
            optimization. Defaults to ``False``.

    Raises:
        RuntimeError: If ``preprocessor`` is used and ``optimizer`` is not passed as
        getter.
    """
    if isinstance(optimizer, Optimizer) and preprocessor:
        raise RuntimeError(
            "If a preprocessor is used, optimizer has to be passed as getter"
        )

    if optimizer is None:
        optimizer = default_image_optimizer

    if isinstance(num_steps, int):
        num_steps = range(num_steps)
    else:
        msg = build_deprecation_message("Passing num_steps iterable of ints", "1.0.0")
        warnings.warn(msg)

    if preprocessor:
        with torch.no_grad():
            input_image = preprocessor(input_image)

    if not isinstance(optimizer, Optimizer):
        optimizer = optimizer(input_image)

    mle_handler = loss_.MLEHandler(criterion)  # type: ignore[attr-defined]

    def closure(input_image: torch.Tensor) -> float:
        # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
        optimizer.zero_grad()  # type: ignore[union-attr]

        with mle_handler(input_image):
            loss = criterion(input_image)
        loss.backward()

        return float(loss)

    with OptimProgressBar("Image optimization", num_steps, quiet=quiet) as progress_bar:
        for _ in num_steps:
            loss = optimizer.step(lambda: closure(input_image))
            progress_bar.update(loss=loss)

    if postprocessor:
        with torch.no_grad():
            input_image = postprocessor(input_image)

    return input_image.detach()


def pyramid_image_optimization(
    input_image: torch.Tensor,
    criterion: nn.Module,
    pyramid: ImagePyramid,
    get_optimizer: Optional[Callable[[torch.Tensor], Optimizer]] = None,
    preprocessor: Optional[nn.Module] = None,
    postprocessor: Optional[nn.Module] = None,
    quiet: bool = False,
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
        quiet: If ``True``, no information is printed to STDOUT during the
            optimization. Defaults to ``False``.
    """
    aspect_ratio = extract_aspect_ratio(input_image)
    if get_optimizer is None:
        get_optimizer = default_image_optimizer

    output_image = input_image
    for level in OptimProgressBar("Pyramid", pyramid, quiet=quiet):
        with torch.no_grad():
            input_image = level.resize_image(output_image, aspect_ratio=aspect_ratio)

        output_image = image_optimization(
            input_image,
            criterion,
            optimizer=get_optimizer,
            num_steps=level.num_steps,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            quiet=quiet,
        )

    return output_image


def default_model_optimizer(transformer: nn.Module) -> Optimizer:
    r"""
    Args:
        transformer: Transformer to be optimized.

    Returns:
        :class:`torch.optim.Adam` optimizer with a learning rate of ``1e-3``. The
        parameters of ``transformer`` are set as optimization parameters.
    """
    return optim.Adam(transformer.parameters(), lr=1e-3)


def unsupervise(image_loader: Iterable) -> Iterator[torch.Tensor]:
    for input in image_loader:
        if isinstance(input, (tuple, list)):
            input = input[0]

        yield input


def model_optimization(
    image_loader: DataLoader,
    transformer: nn.Module,
    criterion: nn.Module,
    criterion_update_fn: Optional[Callable[[torch.Tensor, nn.Module], None]] = None,
    optimizer: Optional[Optimizer] = None,
    quiet: bool = False,
) -> nn.Module:
    r"""Perform a model optimization for a single epoch with integrated logging.

    Args:
        image_loader: Images used as input for the ``transformer``. Drawing from this
            should yield either an batched image or a tuple or list with a batched
            image as first item.
        transformer: Transformer to be optimized.
        criterion: Optimization criterion.
        criterion_update_fn: Is called before each optimization step with the current
            images and the optimization ``criterion``. If omitted and ``criterion`` is
            a :class:`~pystiche.loss.PerceptualLoss` or a
            :class:`~pystiche.loss.GuidedPerceptualLoss` this defaults to invoking
            :meth:`~pystiche.loss.PerceptualLoss.set_content_image`.
        optimizer: Optional optimizer. If ``None``,
            :func:`default_model_optimizer` is used.
        quiet: If ``True``, no information is printed to STDOUT during the
            optimization. Defaults to ``False``.
    """
    if criterion_update_fn is None:
        if isinstance(criterion, loss_.PerceptualLoss):

            def criterion_update_fn(  # type: ignore[misc]
                input_image: torch.Tensor, criterion: loss_.PerceptualLoss,
            ) -> None:
                criterion.set_content_image(input_image)

        else:
            raise RuntimeError(
                f"The parameter 'criterion_update_fn' can only be omitted if the "
                f"'criterion' is a loss.PerceptualLoss or a loss.GuidedPerceptualLoss. "
                f"Got {type(criterion)} instead."
            )

    if optimizer is None:
        optimizer = default_model_optimizer(transformer)

    device = next(transformer.parameters()).device
    mle_handler = loss_.MLEHandler(criterion)  # type: ignore[attr-defined]

    def closure(input_image: torch.Tensor) -> float:
        # See https://github.com/pmeier/pystiche/pull/264#discussion_r430205029
        optimizer.zero_grad()  # type: ignore[union-attr]

        output_image = transformer(input_image)

        with mle_handler(output_image):
            loss = criterion(output_image)
        loss.backward()

        return float(loss)

    with OptimProgressBar(
        "Model optimization", image_loader, quiet=quiet
    ) as progress_bar:
        for input_image in unsupervise(image_loader):
            input_image = input_image.to(device)
            criterion_update_fn(input_image, criterion)  # type: ignore[misc]

            loss = optimizer.step(lambda: closure(input_image))
            progress_bar.update(loss=loss)

    return transformer


def multi_epoch_model_optimization(
    image_loader: DataLoader,
    transformer: nn.Module,
    criterion: nn.Module,
    criterion_update_fn: Optional[Callable[[torch.Tensor, nn.Module], None]] = None,
    epochs: int = 2,
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[LRScheduler] = None,
    quiet: bool = False,
) -> nn.Module:
    r"""Perform a model optimization for multiple epochs with integrated logging.

    Args:
        image_loader: Images used as input for the ``transformer``. Drawing from this
            should yield either an batched image or a tuple or list with a batched
            image as first item.
        transformer: Transformer to be optimized.
        criterion: Optimization criterion.
        criterion_update_fn: Is called before each optimization step with the current
            images and the optimization ``criterion``. If omitted and ``criterion`` is
            a :class:`~pystiche.loss.PerceptualLoss` or a
            :class:`~pystiche.loss.GuidedPerceptualLoss` this defaults to invoking
            :meth:`~pystiche.loss.PerceptualLoss.set_content_image`.
        epochs: Number of epochs. Defaults to ``2``.
        optimizer: Optional optimizer. If ``None``, it is extracted from
            ``lr_scheduler`` or func:`default_model_optimizer` is used.
        lr_scheduler: Optional learning rate scheduler. ``step()`` is invoked after
            every epoch.
        quiet: If ``True``, no information is printed to STDOUT during the
            optimization. Defaults to ``False``.
    """
    if optimizer is None:
        if lr_scheduler is None:
            optimizer = default_model_optimizer(transformer)
        else:
            # Every LRScheduler has optimizer as a valid attribute
            # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            # but this is not reflected in the torch type hints
            optimizer = lr_scheduler.optimizer  # type: ignore[attr-defined]

    for _ in OptimProgressBar("Epochs", epochs, quiet=quiet):
        transformer = model_optimization(
            image_loader,
            transformer,
            criterion,
            criterion_update_fn,
            optimizer,
            quiet=quiet,
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

    return transformer
