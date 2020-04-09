from torch import nn, optim
from torch.optim.optimizer import Optimizer


def sanakoyeu_et_al_2018_optimizer(transformer: nn.Module) -> Optimizer:
    return optim.Adam(transformer.parameters(), lr=2e-4)
