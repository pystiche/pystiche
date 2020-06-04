import contextlib
import logging
import sys
import time
from datetime import datetime
from typing import Callable, Iterator, Optional, Tuple, Union, cast

import torch
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer

import pystiche
from pystiche.pyramid.level import PyramidLevel

from .meter import AverageMeter, ETAMeter, LossMeter, ProgressMeter

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
    INDENT = 2
    SEP_LINE_LENGTH = 80
    SEP_CHARS = ("#", "*", "=", "-", "^", '"')

    def __init__(self, logger: Optional[logging.Logger] = None):
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
        abs_indent = self._calc_abs_indent(indent, rel)
        for line in msg.splitlines():
            self.logger.info(" " * abs_indent + line)

    def sepline(self, level: int = 0, rel: bool = True) -> None:
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
        if top_sep:
            self.sepline(level=level, rel=rel)
        self.message(msg, rel=rel)
        if bottom_sep:
            self.sepline(level=level, rel=rel)

    @contextlib.contextmanager
    def environment(self, header: str) -> Iterator:
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
    def log_fn(step: int, loss: Union[torch.Tensor, pystiche.LossDict]) -> None:
        if step % log_freq == 0:
            with optim_logger.environment(f"Step {step}"):
                if isinstance(loss, pystiche.LossDict):
                    loss = loss.aggregate(max_depth)

                if isinstance(loss, pystiche.LossDict):
                    msg = loss.format()
                elif isinstance(loss, torch.Tensor):
                    msg = f"loss: {loss.item():.3e}"
                else:
                    raise TypeError

                optim_logger.message(msg)

    return log_fn


def default_pyramid_level_header(
    num: int, level: PyramidLevel, input_image_size: Tuple[int, int]
) -> str:
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
    return f"Epoch {epoch}"
