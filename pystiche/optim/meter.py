from typing import Any, Union, Optional, Collection as CollectionType
from abc import ABC, abstractmethod
from collections import deque, OrderedDict, Collection
import pystiche
from pystiche.misc import build_fmtstr, warn_deprecation

__all__ = ["FloatMeter", "LossMeter", "TimeMeter", "ProgressMeter"]


class Meter(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self, arg: Any) -> None:
        pass


# float meter: single float base for average meter / eta meter
# maybe ABC, build ETA first

# average meter: single or multiple floats

# class ProgressMeter(Meter):


class FloatMeter:
    def __init__(
        self,
        name: str,
        fmt: str = "{}",
        show_avg: bool = True,
        use_running_avg: bool = True,
        window_size: int = 10,
    ) -> None:
        self.name = name
        self.fmt = fmt
        self.show_avg = show_avg
        self.use_running_avg = use_running_avg
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.val = 0.0
        self.sum = 0.0
        self.min = 0.0
        self.max = 0.0
        self.window = deque(maxlen=self.window_size)

    @property
    def avg(self) -> float:
        return self.sum / self.count

    @property
    def running_avg(self) -> float:
        vals = tuple(self.window)
        return sum(vals) / len(vals)

    def update(self, vals: Union[CollectionType[float], float]) -> None:
        if isinstance(vals, Collection):
            vals = [float(val) for val in vals]
        else:
            vals = (float(vals),)

        self.count += len(vals)
        self.val = vals[-1]
        self.sum += sum(vals)
        self.window.extend(vals)
        self.min = min(*vals, self.min)
        self.max = max(*vals, self.max)

    def __str__(self) -> str:
        val = self.fmt.format(self.val)
        str = f"{self.name} {val}"
        if self.show_avg:
            avg = self.fmt.format(
                self.running_avg if self.use_running_avg else self.avg
            )
            str += f" ({avg})"
        return str


class AverageMeter(FloatMeter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warn_deprecation(
            "class", "AverageMeter", "0.4", info="Please use FloatMeter instead."
        )
        super().__init__(*args, **kwargs)


class LossMeter(FloatMeter):
    def __init__(self, name: str = "loss", fmt: str = "{:.3e}", **kwargs: Any) -> None:
        super().__init__(name, fmt=fmt, **kwargs)


class TimeMeter(FloatMeter):
    def __init__(self, name: str = "time", fmt: str = "{:3.1f}", **kwargs: Any) -> None:
        warn_deprecation(
            "class", "TimeMeter", "0.4", info="Please use FloatMeter instead."
        )
        super().__init__(name, fmt=fmt, **kwargs)


class ETAMeter(FloatMeter):
    def __init__(self, num_batches: int, name: str = "ETA") -> None:

        pass

    def update(self, val: float) -> None:

        pass


class ProgressMeter(object):
    def __init__(self, num_batches: int, *meters: FloatMeter) -> None:
        self.meters = OrderedDict([(meter.name, meter) for meter in meters])
        self.reset(num_batches=num_batches)

    def reset(self, num_batches: Optional[int] = None) -> None:
        if num_batches is not None:
            fmt = build_fmtstr(field_len=len(str(num_batches)), type="d")
            self._progess_fmt = f"[{fmt}/{num_batches}]"
        self.batch = 0
        for meter in self.meters.values():
            meter.reset()

    @property
    def progress(self) -> str:
        return self._progess_fmt.format(self.batch)

    def update(self, batch: int, **kwargs: Union[Collection[float], float]):
        self.batch = batch
        for name, vals in kwargs.items():
            if name in self.meters:
                self.meters[name].update(vals)

    def __str__(self):
        if not self.meters:
            return self.progress

        return "\t".join(
            [self.progress] + [str(meter) for meter in self.meters.values()]
        )
