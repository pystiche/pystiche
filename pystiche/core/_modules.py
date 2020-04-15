from abc import abstractmethod
from typing import Any, Optional, Sequence, Tuple

import torch
from torch import nn

from pystiche.misc import warn_deprecation

from ._objects import ComplexObject

__all__ = ["Module", "SequentialModule"]


class Module(nn.Module, ComplexObject):
    def __init__(
        self,
        named_children: Optional[Sequence[Tuple[str, nn.Module]]] = None,
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

    def add_named_modules(self, modules: Sequence[Tuple[str, nn.Module]]) -> None:
        if isinstance(modules, dict):
            warn_deprecation(
                "parameter named_modules as ",
                "dict",
                "0.4",
                info="To achieve the same behavior you can pass tuple(modules.items())",
            )
            modules = tuple(modules.items())
        for name, module in modules:
            self.add_module(name, module)

    def add_indexed_modules(self, modules: Sequence[nn.Module]) -> None:
        self.add_named_modules(
            [(str(idx), module) for idx, module in enumerate(modules)]
        )

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __repr__(self):
        return ComplexObject.__repr__(self)

    def torch_repr(self):
        return nn.Module.__repr__(self)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={value}" for key, value in self.properties().items()])


class SequentialModule(Module):
    def __init__(self, *modules: nn.Module):
        super().__init__(indexed_children=modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x
