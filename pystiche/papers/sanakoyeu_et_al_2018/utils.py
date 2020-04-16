from typing import Union
import torch
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from pystiche.optim.meter import FloatMeter
from pystiche.ops.container import OperatorContainer
from pystiche.ops.op import ComparisonOperator


def sanakoyeu_et_al_2018_optimizer(transformer: nn.Module) -> Optimizer:
    return optim.Adam(transformer.parameters(), lr=2e-4)


class ExponentialMovingAverage(FloatMeter):
    def __init__(self, name: str, init_val: float = 0.8, alpha: float = 0.05):
        super().__init__(name)
        self.alpha = alpha
        self.last_val = init_val

    def calculate_val(self, val: Union[torch.tensor, float]):
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.last_val * (1.0 - self.alpha) + self.alpha * val

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
