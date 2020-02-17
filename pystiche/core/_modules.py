from abc import abstractmethod
from typing import Any, Optional, Sequence, Dict
from collections import OrderedDict
import torch
from torch import nn
from ._base import Object

__all__ = ["Module", "SequentialModule"]


class Module(nn.Module, Object):
    def __init__(
        self,
        named_children: Optional[Dict[str, nn.Module]] = None,
        indexed_children: Optional[Sequence[nn.Module]] = None,
    ):
        super().__init__()
        if named_children is not None and indexed_children is not None:
            msg = (
                "named_children and indexed_children "
                "are mutually exclusive parameters."
            )
            raise RuntimeError(msg)
        elif named_children is not None:
            self.add_named_modules(named_children)
        elif indexed_children is not None:
            self.add_indexed_modules(indexed_children)

    def add_named_modules(self, modules: Dict[str, nn.Module]) -> None:
        for name, module in modules.items():
            self.add_module(name, module)

    def add_indexed_modules(self, modules: Sequence[nn.Module]) -> None:
        self.add_named_modules(
            OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])
        )

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={value}" for key, value in self.properties().items()])


class SequentialModule(Module):
    def __init__(self, *modules: nn.Module):
        super().__init__(indexed_children=modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x
