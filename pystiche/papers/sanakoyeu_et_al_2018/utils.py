from torch import nn, optim
from torch.optim.optimizer import Optimizer
from pystiche.optim.meter import FloatMeter


def sanakoyeu_et_al_2018_optimizer(transformer: nn.Module) -> Optimizer:
    return optim.Adam(transformer.parameters(), lr=2e-4)


class ExponentialMovingAverage(FloatMeter):
    def __init__(self, name: str, init_val: float = 0.8, alpha: float = 0.05):
        super().__init__(name)
        self.alpha = alpha
        self.last_val = init_val

    def calculate_val(self, val: float):
        return self.last_val * (1.0 - self.alpha) + self.alpha * val

    def update(self, new_val: float) -> None:
        super().update(self.calculate_val(new_val))

    def local_avg(self) -> float:
        return self.last_val

    def __str__(self) -> str:
        def format(val: float) -> str:
            return self.fmt.format(val)

        val = format(self.last_val)
        info = f"{val}"
        return f"{self.name} {info}"
