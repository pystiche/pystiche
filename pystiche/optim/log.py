import contextlib
import logging
import sys
import time
import warnings
from datetime import datetime
from typing import Callable, Iterator, Optional, Tuple, Union, cast

import torch
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer

import pystiche
from pystiche.misc import build_deprecation_message
from pystiche.pyramid.level import PyramidLevel

from .meter import AverageMeter, ETAMeter, LossMeter, ProgressMeter


def _deprecation_warning() -> None:
    msg = build_deprecation_message(
        "Using any functionality from pystiche.optim.log",
        "0.7.0",
        info="See https://github.com/pmeier/pystiche/issues/434 for details.",
    )
    warnings.warn(msg, UserWarning)


__all__ = [
    "default_logger",
    "OptimLogger",
    "default_image_optim_log_fn",
    "default_pyramid_level_header",
    "default_transformer_optim_log_fn",
    "default_epoch_header",
]


def default_logger(
    name: Optional[str] = None, log_file: Optional[str] = None
) -> logging.Logger:
    r"""Logs :attr:`logging.INFO` to :class:`sys.stdout` and all log levels above to
    :class:`sys.stderr`. Each log entry is timestamped and can optionally be written to
    a log_file.

    Args:
        name: Optional name. See :class:`logging.Logger` for details.
        log_file: Optional path to log file.
    """
    _deprecation_warning()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        fmt="|%(asctime)s| %(message)s", datefmt="%d.%m.%Y %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.addFilter(lambda record: record.levelno <= logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class OptimLogger:
    r"""Hierarchical optimization logger.

    Args:
        logger: Optional :class:`logging.Logger` that is used. Defaults to
            :func:`default_logger` with a unique name.
    """
    INDENT = 2
    SEP_LINE_LENGTH = 80
    SEP_CHARS = ("#", "*", "=", "-", "^", '"')

    def __init__(self, logger: Optional[logging.Logger] = None):
        _deprecation_warning()
        if logger is None:
            name = f"pystiche_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            logger = default_logger(name=name)
        self.logger = logger

        self._environ_indent_offset = 0
        self._environ_level_offset = 0

    def _calc_abs_indent(self, indent: int, rel: bool) -> int:
        abs_indent = indent
        if rel:
            abs_indent += self._environ_indent_offset
        return abs_indent

    def _calc_abs_level(self, level: int, rel: bool) -> int:
        abs_level = level
        if rel:
            abs_level += self._environ_level_offset
        return abs_level

    def message(self, msg: str, indent: int = 0, rel: bool = True) -> None:
        r"""Log a message with optional indent.

        Args:
            msg: Message.
            indent: Optional indent. Defaults to ``0``.
            rel: If ``True``, the indentation is determined relative to the current
                level. Defaults to ``True``.
        """
        abs_indent = self._calc_abs_indent(indent, rel)
        for line in msg.splitlines():
            self.logger.info(" " * abs_indent + line)

    def sepline(self, level: int = 0, rel: bool = True) -> None:
        r"""Log a separation line.

        Args:
            level: Determines the separation char following
                `Python's Style Guide for documenting <https://devguide.python.org/documenting/#sections>`_
                .
            rel: If ``True``, the level is determined relative to the current level.
                Defaults to ``True``.
        """
        abs_level = self._calc_abs_level(level, rel)
        self.message(self.SEP_CHARS[abs_level] * self.SEP_LINE_LENGTH)

    def sep_message(
        self,
        msg: str,
        level: int = 0,
        rel: bool = True,
        top_sep: bool = True,
        bottom_sep: bool = True,
    ) -> None:
        r"""Log a message with optional enclosing separation lines.

        Args:
            msg: Message.
            level: Determines the separation char following
                `Python's Style Guide for documenting <https://devguide.python.org/documenting/#sections>`_
                .
            rel: If ``True``, the level is determined relative to the current level.
                Defaults to ``True``.
            top_sep: If ``True``, add a separation line before the message. Defaults to
                ``True``.
            bottom_sep: If ``True``, add a separation line after the message. Defaults
                to ``True``.
        """
        if top_sep:
            self.sepline(level=level, rel=rel)
        self.message(msg, rel=rel)
        if bottom_sep:
            self.sepline(level=level, rel=rel)

    @contextlib.contextmanager
    def environment(self, header: Optional[str]) -> Iterator:
        r"""Context manager that increases the current level. Can be nested.

        Args:
            header: Optional header that is logged with :meth:`.sep_message`.
        """
        if header is not None:
            self.sep_message(header)
        self._environ_indent_offset += self.INDENT
        self._environ_level_offset += 1
        try:
            yield
        finally:
            self._environ_level_offset -= 1
            self._environ_indent_offset -= self.INDENT


def default_image_optim_log_fn(
    optim_logger: OptimLogger, log_freq: int = 50, max_depth: int = 1
) -> Callable[[int, Union[torch.Tensor, pystiche.LossDict]], None]:
    r"""Log the loss during image optimizations.

    Args:
        optim_logger: Optimization logger.
        log_freq: Number of optimization steps between two log entries. Defaults to
            ``50``.
        max_depth: If loss is :class:`pystiche.LossDict`, aggregate it to this maximum
            depth before logging. Defaults to ``1``.
    """
    _deprecation_warning()

    def log_fn(step: int, loss: Union[torch.Tensor, pystiche.LossDict]) -> None:
        if step % log_freq == 0:
            with optim_logger.environment(f"Step {step}"):
                if isinstance(loss, pystiche.LossDict):
                    loss = loss.aggregate(max_depth)

                if isinstance(loss, pystiche.LossDict):
                    msg = str(loss)
                elif isinstance(loss, torch.Tensor):
                    msg = f"loss: {loss.item():.3e}"
                else:
                    msg = (  # type: ignore[unreachable]
                        f"loss can be a pystiche.LossDict or a scalar torch.Tensor, "
                        f"but got {type(loss)}."
                    )
                    raise TypeError(msg)

                optim_logger.message(msg)

    return log_fn


def default_pyramid_level_header(
    num: int, level: PyramidLevel, input_image_size: Tuple[int, int]
) -> str:
    r"""
    Args:
        num: Number of the pyramid level.
        level: Pyramid level.
        input_image_size: Image size of the input image.

    Returns:
        Header that includes information about the current level.
    """
    _deprecation_warning()
    height, width = input_image_size
    return f"Pyramid level {num} with {level.num_steps} steps " f"({width} x {height})"


def default_transformer_optim_log_fn(
    optim_logger: OptimLogger,
    num_batches: int,
    log_freq: Optional[int] = None,
    show_loading_velocity: bool = True,
    show_processing_velocity: bool = True,
    show_running_means: bool = True,
) -> Callable[[int, Union[float, torch.Tensor, pystiche.LossDict], float, float], None]:
    r"""Log the ETA, the loss, and the loading and processing velocities during
    transformer optimizations.

    Args:
        optim_logger: Optimization logger.
        num_batches: Total number of batches.
        log_freq: Number of batches between two log entries. If ``None`` a sensible
            value is chosen based on ``num_batches``. Defaults to ``None``.
        show_loading_velocity: If ``True``, includes the image loading velocity in the
            log entries. Defaults to ``True``.
        show_processing_velocity: If ``True``, includes the image processing velocity
            in the log entries. Defaults to ``True``.
        show_running_means: If ``True``, includes a running mean additional to the
            current values in the log entries. The window size is chosen based on
            ``log_freq``. Defaults to ``True``.
    """
    _deprecation_warning()
    if log_freq is None:
        log_freq = max(min(round(1e-3 * num_batches) * 10, 50), 1)

    window_size = min(10 * log_freq, 1000)

    meters = [
        ETAMeter(
            num_batches, window_size=window_size, show_local_eta=show_running_means
        ),
        LossMeter(window_size=window_size, show_local_avg=show_running_means),
    ]
    if show_loading_velocity:
        meters.append(
            AverageMeter(
                name="loading",
                window_size=window_size,
                show_local_avg=show_running_means,
                fmt="{:4.0f} img/s",
            )
        )
    if show_processing_velocity:
        meters.append(
            AverageMeter(
                name="processing",
                window_size=window_size,
                show_local_avg=show_running_means,
                fmt="{:4.0f} img/s",
            )
        )

    progress_meter = ProgressMeter(num_batches, *meters)

    def log_fn(
        batch: int,
        loss: Union[float, torch.Tensor, pystiche.LossDict],
        loading_velocity: float,
        processing_velocity: float,
    ) -> None:
        progress_meter.update(
            ETA=time.time(),
            loss=loss,
            loading=loading_velocity,
            processing=processing_velocity,
        )
        if batch % cast(int, log_freq) == 0:
            optim_logger.message(str(progress_meter))

    return log_fn


def default_epoch_header(
    epoch: int, optimizer: Optimizer, lr_scheduler: Optional[LRScheduler]
) -> str:
    r"""
    Args:
        epoch: Number of the epoch.
        optimizer: Optimizer.
        lr_scheduler: Optional learning rate scheduler.

    Returns:
        Header that includes information about the current epoch.
    """
    _deprecation_warning()
    return f"Epoch {epoch}"
