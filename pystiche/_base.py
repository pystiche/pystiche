from typing import Any, Optional, Sequence, Tuple, Dict, Iterator, NoReturn, Union
from abc import ABC, abstractmethod
from copy import copy
from collections import OrderedDict
import torch
from torch import nn
from .misc import build_obj_str, build_fmtstr, format_dict


__all__ = ["Object", "Module", "TensorStorage", "LossDict"]


class Object(ABC):
    _STR_INDENT = 2

    def _properties(self) -> Dict[str, Any]:
        return OrderedDict()

    def extra_properties(self) -> Dict[str, Any]:
        return OrderedDict()

    def properties(self) -> Dict[str, Any]:
        dct = self._properties()
        dct.update(self.extra_properties())
        return dct

    def _named_children(self) -> Iterator[Tuple[str, Any]]:
        return
        yield

    def extra_named_children(self) -> Iterator[Tuple[str, Any]]:
        return
        yield

    def named_children(self) -> Iterator[Tuple[str, Any]]:
        yield from self._named_children()
        yield from self.extra_named_children()

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


class Module(nn.Module, Object):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Any:
        pass

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={value}" for key, value in self.properties().items()])


class TensorStorage(nn.Module, Object):
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


class LossDict(OrderedDict):
    def __init__(
        self,
        losses: Optional[Sequence[Tuple[str, Union[torch.Tensor, "LossDict"]]]] = None,
    ):
        unrolled_losses = []
        for name, loss in losses:
            if isinstance(loss, torch.Tensor):
                unrolled_losses.append((name, loss))
            else:
                for child_name, child_loss in loss.items():
                    unrolled_losses.append((f"{name}.{child_name}", child_loss))
        super().__init__(unrolled_losses)

    def aggregate(self, max_depth):
        if max_depth == 0:
            return sum(self.values())

        splits = [name.split(".") for name in self.keys()]
        if not any([len(split) >= max_depth for split in splits]):
            return copy(self)

        agg_names = [".".join(split[:max_depth]) for split in splits]
        key_map = dict(zip(self.keys(), agg_names))
        agg_losses = {name: [] for name in set(agg_names)}
        for name, loss in self.items():
            agg_losses[key_map[name]].append(loss)

        return LossDict([(name, sum(agg_losses[name])) for name in agg_names])

    def total(self) -> torch.Tensor:
        return self.aggregate(0)

    def backward(self, *args, **kwargs) -> None:
        self.total().backward(*args, **kwargs)

    def item(self):
        return self.total().item()

    def __float__(self):
        return self.item()

    def __mul__(self, other):
        return LossDict([(name, loss * other) for name, loss in self.items()])

    def format(self, max_depth=None, **format_dict_kwargs):
        if max_depth is not None:
            dct = self.aggregate(max_depth)
        else:
            dct = self

        fmt = build_fmtstr(precision=3, type="e")
        values = [fmt.format(value.item()) for value in dct.values()]
        return format_dict(OrderedDict(zip(dct.keys(), values)), **format_dict_kwargs)

    def __str__(self):
        return self.format()
