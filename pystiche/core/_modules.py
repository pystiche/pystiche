from abc import abstractmethod
from typing import Any, Dict
from torch import nn
from ._base import Object


class Module(nn.Module, Object):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Any:
        pass

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={value}" for key, value in self.properties().items()])
