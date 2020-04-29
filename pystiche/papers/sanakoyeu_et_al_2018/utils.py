from typing import Any, List, Optional, Union

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer

from pystiche.ops.container import OperatorContainer
from pystiche.ops.op import ComparisonOperator
from pystiche.optim.meter import FloatMeter


def sanakoyeu_et_al_2018_optimizer(
    params: Union[nn.Module, List[torch.Tensor]]
) -> Optimizer:
    if isinstance(params, nn.Module):
        params = params.parameters()
    return optim.Adam(params, lr=2e-4)


def sanakoyeu_et_al_2018_lr_scheduler(optimizer: Optimizer) -> Optional[ExponentialLR]:
    return DelayedExponentialLR(optimizer, gamma=0.1, delay=2)


class ExponentialMovingAverage(FloatMeter):
    def __init__(
        self, name: str, init_val: float = 0.8, smoothing_factor: float = 0.05
    ):
        super().__init__(name)
        self.smoothing_factor = smoothing_factor
        self.last_val = init_val

    def calculate_val(self, val: Union[torch.tensor, float]):
        if isinstance(val, torch.Tensor):
            val = val.item()
        return (
            self.last_val * (1.0 - self.smoothing_factor) + self.smoothing_factor * val
        )

    def update(self, new_val: Union[torch.tensor, float]) -> None:
        super().update(self.calculate_val(new_val))

    def local_avg(self) -> float:
        return self.last_val

    def __str__(self) -> str:
        def format(val: float) -> str:
            return self.fmt.format(val)

        val = format(self.last_val)
        info = f"{val}"
        return f"{self.name} {info}"


class ContentOperatorContainer(OperatorContainer):
    def set_target_image(self, image: torch.Tensor):
        for op in self.children():
            if isinstance(op, ComparisonOperator):
                op.set_target_image(image)


class DelayedExponentialLR(
    ExponentialLR
):  # TODO: move to optim.lr_scheduler? used in two paper
    def __init__(
        self, optimizer: Optimizer, gamma: float, delay: int, **kwargs: Any
    ) -> None:
        self.delay = delay
        super().__init__(optimizer, gamma, **kwargs)

    def get_lr(self) -> List[float]:
        exp = self.last_epoch - self.delay + 1
        if exp > 0:
            return [base_lr * self.gamma ** exp for base_lr in self.base_lrs]
        else:
            return self.base_lrs
