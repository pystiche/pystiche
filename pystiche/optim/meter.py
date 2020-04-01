from typing import Any, Union, Optional, Collection, Callable
from abc import ABC, abstractmethod
from collections import deque, OrderedDict
from datetime import datetime, timedelta
import torch
from pystiche.misc import build_fmtstr, build_deprecation_message, warn_deprecation

__all__ = [
    "Meter",
    "FloatMeter",
    "AverageMeter",
    "LossMeter",
    "ETAMeter",
    "ProgressMeter",
]


class Meter(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class FloatMeter(Meter):
    def __init__(self, name: str, window_size: int = 50) -> None:
        super().__init__(name)
        self.window_size = window_size

        self.count = None
        self.last_val = None
        self.global_sum = None
        self.global_min = None
        self.global_max = None
        self.window = None
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.last_val = 0.0
        self.global_sum = 0.0
        self.global_min = 0.0
        self.global_max = 0.0
        self.window = deque(maxlen=self.window_size)

    def update(self, vals: Union[Collection[float], float]) -> None:
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
        return self.global_sum / self.count

    @property
    def local_avg(self) -> float:
        vals = tuple(self.window)
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
        show_avg: Optional[bool] = None,
        use_running_avg: Optional[bool] = None,
    ):
        if show_avg is not None:
            msg = build_deprecation_message(
                "parameter", "show_avg", "0.4", info="The average is now always shown."
            )
            if show_avg:
                warn_deprecation(msg)
            else:
                raise RuntimeError(msg)

        if use_running_avg is not None:
            warn_deprecation(
                "parameter",
                "use_running_avg",
                "0.4",
                info="It was renamed to show_local_avg.",
            )
            show_local_avg = use_running_avg

        super().__init__(name=name, window_size=window_size)
        self.show_local_avg = show_local_avg
        self.fmt = fmt

    def update(
        self, vals: Union[Collection[float], float, torch.Tensor],
    ):
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().flatten().tolist()

        super().update(vals)

    def __str__(self) -> str:
        def format(val: float) -> str:
            return self.fmt.format(val)

        val = format(self.last_val)
        avg = format(self.local_avg if self.show_local_avg else self.global_avg)
        return f"{self.name} {val} ({avg})"

    @property
    def val(self) -> float:
        warn_deprecation("attribute", "val", "0.4", info="It was renamed to last_val")
        return self.global_avg

    @property
    def avg(self) -> float:
        warn_deprecation("attribute", "avg", "0.4", info="It was renamed to global_avg")
        return self.global_avg

    @property
    def min(self) -> float:
        warn_deprecation("attribute", "min", "0.4", info="It was renamed to global_min")
        return self.global_avg

    @property
    def max(self) -> float:
        warn_deprecation("attribute", "max", "0.4", info="It was renamed to global_max")
        return self.global_avg

    @property
    def running_avg(self) -> float:
        warn_deprecation(
            "attribute", "running_avg", "0.4", info="It was renamed to local_avg"
        )
        return self.local_avg


class LossMeter(AverageMeter):
    def __init__(self, name: str = "loss", fmt: str = "{:.3e}", **kwargs: Any) -> None:
        super().__init__(name, fmt=fmt, **kwargs)


class TimeMeter(AverageMeter):
    def __init__(self, name: str = "time", fmt: str = "{:3.1f}", **kwargs: Any) -> None:
        warn_deprecation(
            "class", "TimeMeter", "0.4", info="Please use AverageMeter instead."
        )
        super().__init__(name, fmt=fmt, **kwargs)


class ETAMeter(FloatMeter):
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

        self.last_time = None
        self.reset()

    def reset(self):
        super().reset()
        self.last_time = None

    def update(self, time: float):
        if self.last_time is None:
            self.last_time = time
            return

        time_diff = time - self.last_time
        super().update(time_diff)
        self.last_time = time

    def calculate_eta(self, time_diff: float) -> str:
        count_diff = max(self.total_count - self.count, 0)
        now = datetime.now()
        if count_diff <= 0:
            return now.strftime(self.fmt)

        eta = now + count_diff * timedelta(seconds=time_diff)
        return eta.strftime(self.fmt)

    @property
    def global_eta(self) -> str:
        return self.calculate_eta(self.global_avg)

    @property
    def local_eta(self) -> str:
        return self.calculate_eta(self.local_avg)

    def __str__(self):
        if self.count > 0:
            eta = self.local_eta if self.show_local_eta else self.global_eta
        else:
            eta = "N/A"
        return f"{self.name} {eta}"


class ProgressMeter(Meter):
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

        self.build_progress_fmt = build_progress_fmt

        self.count = None
        self.progress_fmt = None
        self.reset(total_count=total_count)

    def reset(
        self, total_count: Optional[int] = None, num_batches: Optional[int] = None
    ) -> None:
        if num_batches is not None:
            warn_deprecation(
                "parameter", "num_batches", "0.4", info="It was renamed to total_count."
            )
            total_count = num_batches

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

    def __str__(self):
        parts = []

        if self.name is not None:
            parts.append(self.name)

        parts.append(self.progress)

        if self.meters:
            parts.extend([str(meter) for meter in self.meters.values()])

        return "\t".join(parts)
