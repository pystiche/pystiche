from collections import deque, OrderedDict
from pystiche.misc import build_fmtstr


__all__ = ["AverageMeter", "LossMeter", "TimeMeter", "ProgressMeter"]


class AverageMeter:
    def __init__(
        self,
        name: str,
        fmt="{}",
        show_avg=True,
        use_running_avg=True,
        window_size: int = 10,
    ):
        self.name = name
        self.fmt = fmt
        self.show_avg = show_avg
        self.use_running_avg = use_running_avg
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.count = 0
        self.val = 0.0
        self.avg = 0.0
        self.window = deque(maxlen=self.window_size)
        self.min = 0.0
        self.max = 0.0

    @property
    def running_avg(self):
        vals = tuple(self.window)
        return sum(vals) / len(vals)

    def update(self, val: float):
        self.count += 1
        self.val = val
        self.avg += val  # FIXME
        self.window.append(val)
        self.min = min(val, self.min)
        self.max = max(val, self.max)

    def __str__(self):
        val = self.fmt.format(self.val)
        str = f"{self.name} {val}"
        if self.show_avg:
            avg = self.fmt.format(
                self.running_avg if self.use_running_avg else self.avg
            )
            str += f" ({avg})"
        return str


class LossMeter(AverageMeter):
    def __init__(self, name="loss", fmt="{:.3e}", **kwargs):
        super().__init__(name, fmt=fmt, **kwargs)

    def update(self, val: float):
        super().update(float(val))


class TimeMeter(AverageMeter):
    def __init__(self, name="time", fmt="{:3.1f}", **kwargs):
        super().__init__(name, fmt=fmt, **kwargs)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.meters = OrderedDict([(meter.name, meter) for meter in meters])
        self.reset(num_batches=num_batches)

    def reset(self, num_batches=None):
        if num_batches is not None:
            fmt = build_fmtstr(field_len=len(str(num_batches)), type="d")
            self._progess_fmt = f"[{fmt}/{num_batches}]"
        self.batch = 0
        for meter in self.meters.values():
            meter.reset()

    @property
    def progress(self):
        return self._progess_fmt.format(self.batch)

    def update(self, batch, **kwargs):
        self.batch = batch
        for name, val in kwargs.items():
            if name in self.meters:
                self.meters[name].update(val)

    def __str__(self):
        return "\t".join(
            [self.progress] + [str(meter) for meter in self.meters.values()]
        )
