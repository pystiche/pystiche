import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Sequence, Union, cast

import torch

import pystiche
from pystiche.misc import build_deprecation_message, build_fmtstr

__all__ = [
    "Meter",
    "FloatMeter",
    "AverageMeter",
    "LossMeter",
    "ETAMeter",
    "ProgressMeter",
]


def _deprecation_warning() -> None:
    msg = build_deprecation_message(
        "Using any functionality from pystiche.optim.meter",
        "0.7.0",
        info="See https://github.com/pmeier/pystiche/issues/434 for details.",
    )
    warnings.warn(msg, UserWarning)


class Meter(ABC):
    def __init__(self, name: Optional[str] = None):
        _deprecation_warning()
        self.name = name

    @abstractmethod
    def reset(self) -> None:
        pass

    update: Callable[..., None]

    @abstractmethod
    def __str__(self) -> str:
        pass


class FloatMeter(Meter):
    count: int
    last_val: float
    global_sum: float
    global_min: float
    global_max: float
    window: deque

    def __init__(self, name: str, window_size: int = 50) -> None:
        super().__init__(name)
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.last_val = 0.0
        self.global_sum = 0.0
        self.global_min = float("inf")
        self.global_max = -float("inf")
        self.window = deque(maxlen=self.window_size)

    def update(self, vals: Union[Sequence[float], float]) -> None:
        if isinstance(vals, float):
            vals = (vals,)

        self.count += len(vals)
        self.last_val = vals[-1]
        self.global_sum += sum(vals)
        self.global_min = min(*vals, self.global_min)
        self.global_max = max(*vals, self.global_max)
        self.window.extend(vals)

    @property
    def global_avg(self) -> float:
        if self.count == 0:
            msg = "global_avg can only be calculated after the first update."
            raise RuntimeError(msg)
        return self.global_sum / self.count

    @property
    def local_avg(self) -> float:
        vals = tuple(self.window)
        if not vals:
            msg = "local_avg can only be calculated after the first update."
            raise RuntimeError(msg)
        return sum(vals) / len(vals)

    @abstractmethod
    def __str__(self) -> str:
        pass


class AverageMeter(FloatMeter):
    def __init__(
        self,
        name: str,
        window_size: int = 50,
        show_local_avg: bool = True,
        fmt: str = "{:f}",
    ):
        super().__init__(name=name, window_size=window_size)
        self.show_local_avg = show_local_avg
        self.fmt = fmt

    def update(self, vals: Union[Sequence[float], float, torch.Tensor]) -> None:
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().flatten().tolist()

        super().update(vals)

    def __str__(self) -> str:
        def format(val: float) -> str:
            return self.fmt.format(val)

        if self.count > 0:
            val = format(self.last_val)
            avg = format(self.local_avg if self.show_local_avg else self.global_avg)
            info = f"{val} ({avg})"
        else:
            info = "N/A"
        return f"{self.name} {info}"


class LossMeter(AverageMeter):
    def __init__(self, name: str = "loss", fmt: str = "{:.3e}", **kwargs: Any) -> None:
        super().__init__(name, fmt=fmt, **kwargs)

    def update(
        self, val: Union[Sequence[float], float, torch.Tensor, pystiche.LossDict],
    ) -> None:
        if isinstance(val, pystiche.LossDict):
            val = float(val)
        super().update(val)


class ETAMeter(FloatMeter):
    last_time: Optional[float]

    def __init__(
        self,
        total_count: int,
        name: str = "ETA",
        window_size: int = 500,
        show_local_eta: bool = True,
        fmt: str = "%d.%m.%Y %H:%M",
    ):
        super().__init__(name=name, window_size=window_size)
        self.total_count = total_count
        self.show_local_eta = show_local_eta
        self.fmt = fmt

        self.reset()

    def reset(self) -> None:
        super().reset()
        self.last_time = None

    # This violates the Liskov Substitution Principle. We ignore this for now since
    # this complete module should be refactored in the future.
    def update(self, time: float) -> None:  # type: ignore[override]
        if self.last_time is None:
            self.last_time = time
            return

        time_diff = time - self.last_time
        super().update(time_diff)
        self.last_time = time

    def calculate_eta(self, time_diff: float) -> datetime:
        count_diff = max(self.total_count - self.count, 0)
        return datetime.now() + count_diff * timedelta(seconds=time_diff)

    @property
    def global_eta(self) -> datetime:
        return self.calculate_eta(self.global_avg)

    @property
    def local_eta(self) -> datetime:
        return self.calculate_eta(self.local_avg)

    def __str__(self) -> str:
        if self.count > 0:
            eta = self.local_eta if self.show_local_eta else self.global_eta
            info = eta.strftime(self.fmt)
        else:
            info = "N/A"
        return f"{self.name} {info}"


class ProgressMeter(Meter):
    count: int
    progress_fmt: str

    def __init__(
        self,
        total_count: int,
        *meters: Meter,
        name: Optional[str] = None,
        build_progress_fmt: Optional[Callable[[int], str]] = None,
    ) -> None:
        super().__init__(name=name)
        self.meters = OrderedDict([(meter.name, meter) for meter in meters])

        if build_progress_fmt is None:

            def build_progress_fmt(total_count: int) -> str:
                count_fmt = build_fmtstr(field_len=len(str(total_count)), type="d")
                return f"[{count_fmt}/{total_count:d}]"

            build_progress_fmt = cast(Callable[[int], str], build_progress_fmt)

        self.build_progress_fmt = build_progress_fmt

        self.reset(total_count=total_count)

    def reset(self, total_count: Optional[int] = None,) -> None:
        self.count = 0

        if total_count is not None:
            self.progress_fmt = self.build_progress_fmt(total_count)

        for meter in self.meters.values():
            meter.reset()

    def update(self, **kwargs: Any) -> None:
        self.count += 1

        for name, vals in kwargs.items():
            self.meters[name].update(vals)

    @property
    def progress(self) -> str:
        return self.progress_fmt.format(self.count)

    def __str__(self) -> str:
        parts = []

        if self.name is not None:
            parts.append(self.name)

        parts.append(self.progress)

        if self.meters:
            parts.extend([str(meter) for meter in self.meters.values()])

        return "\t".join(parts)
