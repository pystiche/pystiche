import warnings
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from pystiche.misc import build_deprecation_message

from ._objects import ComplexObject

__all__ = ["Module", "SequentialModule"]


class Module(nn.Module, ComplexObject):
    r""":class:`torch.nn.Module` with the enhanced representation options of
    :class:`pystiche.ComplexObject`.

    Args:
        named_children: Optional children modules that are added by their names.
        indexed_children: Optional children modules that are added with successive
            indices as names.

    .. note::

        ``named_children`` and ``indexed_children`` are mutually exclusive parameters.
    """
    _buffers: Dict[str, torch.Tensor]
    _modules: Dict[str, nn.Module]

    def __init__(
        self,
        named_children: Optional[Iterable[Tuple[str, nn.Module]]] = None,
        indexed_children: Optional[Sequence[nn.Module]] = None,
    ):
        super().__init__()
        if named_children is not None and indexed_children is not None:
            msg = (
                "named_children and indexed_children "
                "are mutually exclusive parameters."
            )
            raise RuntimeError(msg)
        if indexed_children is not None:
            named_children = [
                (str(idx), module) for idx, module in enumerate(indexed_children)
            ]
        if named_children is not None:
            for name, module in named_children:
                self.add_module(name, module)

    def __repr__(self) -> str:
        return ComplexObject.__repr__(self)

    def torch_repr(self) -> str:
        r"""
        Returns:
            Native torch representation.
        """
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
