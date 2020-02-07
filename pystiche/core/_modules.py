from abc import abstractmethod
from typing import Any, Union, Dict
from collections import OrderedDict
import torch
from torch import nn
from ._base import Object

__all__ = ["Module", "ContainerModule", "SequentialModule"]


class Module(nn.Module, Object):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={value}" for key, value in self.properties().items()])


class ContainerModule(Module):
    def __init__(self, *args: Union[nn.Module, Dict[str, nn.Module]]) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass


class SequentialModule(ContainerModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x
