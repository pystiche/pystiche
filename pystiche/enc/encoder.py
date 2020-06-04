from abc import abstractmethod
from typing import Sequence

import torch
from torch import nn

import pystiche

from .guides import propagate_guide

__all__ = ["Encoder", "SequentialEncoder"]


class Encoder(pystiche.Module):
    r"""ABC for all encoders. Invokes :meth:`Encoder.forward` if called."""

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Encodes the given input.

        .. note::

            This method has to be overwritten in every subclass.
        """
        pass

    @abstractmethod
    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        r"""Encodes the given guide.

        .. note::

            This method has to be overwritten in every subclass.
        """
        pass


class SequentialEncoder(Encoder):
    r"""Encoder that operates in sequential manner. Invokes :meth:`Encoder.forward`
    if called.

    Args:
        modules: Sequential modules.
    """

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__(indexed_children=modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            input = module(input)
        return input

    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            guide = propagate_guide(module, guide)
        return guide
