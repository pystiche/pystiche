from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple, Dict, NoReturn
from collections import OrderedDict
import torch
from torch import nn
from .misc import build_obj_str, to_engstr


class Module(ABC, nn.Module):
    _STR_INDENT = 2

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Any:
        pass

    def _properties(self) -> Dict[str, Any]:
        return OrderedDict()

    def extra_properties(self) -> Dict[str, Any]:
        return OrderedDict()

    def properties(self) -> Dict[str, Any]:
        dct = self._properties()
        dct.update(self.extra_properties())
        return dct

    def _build_str(
        self,
        name: Optional[str] = None,
        properties: Optional[Dict[str, str]] = None,
        named_children: Optional[Sequence[Tuple[str, Any]]] = None,
    ) -> str:
        if name is None:
            name = self.__class__.__name__

        if properties is None:
            properties = self.properties()

        if named_children is None:
            named_children = tuple(self.named_children())

        return build_obj_str(
            name,
            properties=properties,
            named_children=named_children,
            num_indent=self._STR_INDENT,
        )

    def __str__(self) -> str:
        return self._build_str()

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={value}" for key, value in self.properties().items()])


class TensorStorage(nn.Module):
    def __init__(self, **attrs: Dict[str, Any]) -> None:
        super().__init__()
        for name, attr in attrs.items():
            if isinstance(attr, torch.Tensor):
                self.register_buffer(name, attr)
            else:
                setattr(self, name, attr)

    def forward(self) -> NoReturn:
        msg = (
            f"{self.__class__.__name__} objects are only used "
            "for storage and cannot be called."
        )
        raise RuntimeError(msg)
