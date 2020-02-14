from abc import abstractmethod
from typing import Any, Sequence, Dict
from collections import OrderedDict
import torch
from torch import nn
from ._base import Object

__all__ = ["Module", "SequentialModule"]


class Module(nn.Module, Object):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={value}" for key, value in self.properties().items()])

    def add_named_modules(self, modules: Dict[str, nn.Module]) -> None:
        for name, module in modules.items():
            self.add_module(name, module)

    def add_indexed_modules(self, modules: Sequence[nn.Module]) -> None:
        self.add_named_modules(
            OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])
        )


class SequentialModule(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x
