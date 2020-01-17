import torch
from collections import OrderedDict

__all__ = ["LossDict"]


class LossDict(OrderedDict):
    @property
    def total_loss(self) -> torch.Tensor:
        return sum(self.values())

    def backward(self, *args, **kwargs) -> None:
        self.total_loss.backward(*args, **kwargs)

    def __float__(self):
        return self.total_loss.item()
