from collections import OrderedDict
import torch
from pystiche.misc import build_fmtstr, format_dict

__all__ = ["LossDict"]


class LossDict(OrderedDict):
    @property
    def total_loss(self) -> torch.Tensor:
        return sum(self.values())

    def backward(self, *args, **kwargs) -> None:
        self.total_loss.backward(*args, **kwargs)

    def __float__(self):
        return self.total_loss.item()

    def item(self):
        return float(self)

    def __str__(self):
        fmtstr = build_fmtstr(precision=3, type="e")
        values = [fmtstr.format(value.item()) for value in self.values()]
        dct = OrderedDict(zip(self.keys(), values))
        return format_dict(dct)
