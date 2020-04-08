from abc import abstractmethod
from typing import Sequence

import torch
from torch import nn

import pystiche

from .guides import propagate_guide

__all__ = ["Encoder", "SequentialEncoder"]


class Encoder(pystiche.Module):
    @abstractmethod
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        pass


class SequentialEncoder(Encoder):
    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__(indexed_children=modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x

    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            guide = propagate_guide(module, guide)
        return guide
