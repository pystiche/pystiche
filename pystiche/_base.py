from abc import abstractmethod
from typing import Any, Dict, NoReturn
import torch
from torch import nn


class Module(nn.Module):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Any:
        pass


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
