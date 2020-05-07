import warnings
from abc import ABC
from collections import OrderedDict
from copy import copy
from typing import (
    Any,
    Dict,
    Hashable,
    Iterator,
    List,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import nn

from pystiche.meta import is_scalar_tensor
from pystiche.misc import (
    build_complex_obj_repr,
    build_deprecation_message,
    build_fmtstr,
    format_dict,
)

__all__ = ["ComplexObject", "TensorStorage", "LossDict", "TensorKey"]


class ComplexObject(ABC):
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

    def _build_repr(
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

        return build_complex_obj_repr(
            name,
            properties=properties,
            named_children=named_children,
            num_indent=self._STR_INDENT,
        )

    def __repr__(self) -> str:
        return self._build_repr()


class Object(ComplexObject):
    def __init__(self, *args, **kwargs):
        msg = build_deprecation_message(
            "The class Object", "0.4", info="It was renamed to ComplexObject."
        )
        warnings.warn(msg)
        super().__init__(*args, **kwargs)


# TODO: can this be removed for now?
#  If not it should subclass pystiche.Module and thus should be moved to ._modules
class TensorStorage(nn.Module, ComplexObject):
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


from typing import cast, Iterable


class LossDict(OrderedDict):
    def __init__(
        self, losses: Sequence[Tuple[str, Union[torch.Tensor, "LossDict"]]] = (),
    ) -> None:
        super().__init__()
        for name, loss in losses:
            self[name] = loss

    def __setitem__(self, name: str, loss: Union[torch.Tensor, "LossDict"]) -> None:
        if isinstance(loss, torch.Tensor):
            if not is_scalar_tensor(loss):
                # FIXME
                raise TypeError
            super().__setitem__(name, loss)
        elif isinstance(loss, LossDict):
            for child_name, child_loss in loss.items():
                super().__setitem__(f"{name}.{child_name}", child_loss)
        else:
            # FIXME
            raise TypeError

    def aggregate(self, max_depth: int) -> Union[torch.Tensor, "LossDict"]:
        def sum(partial_losses: Iterable[torch.Tensor]):
            return cast(torch.Tensor, sum(partial_losses))

        if max_depth == 0:
            return sum(self.values())

        splits = [name.split(".") for name in self.keys()]
        if not any([len(split) >= max_depth for split in splits]):
            return copy(self)

        agg_names = [".".join(split[:max_depth]) for split in splits]
        key_map = dict(zip(self.keys(), agg_names))
        agg_losses: Dict[str, List[torch.Tensor]] = {
            name: [] for name in set(agg_names)
        }
        for name, loss in self.items():
            agg_losses[key_map[name]].append(loss)

        return LossDict([(name, sum(agg_losses[name])) for name in agg_names])

    def total(self) -> torch.Tensor:
        return cast(torch.Tensor, self.aggregate(0))

    def backward(self, *args, **kwargs) -> None:
        self.total().backward(*args, **kwargs)

    def item(self) -> float:
        return self.total().item()

    def __float__(self) -> float:
        return self.item()

    def __mul__(self, other) -> "LossDict":
        other = float(other)
        return LossDict([(name, loss * other) for name, loss in self.items()])

    # TODO: can this be moved in __str__?
    def format(self, max_depth: Optional[int] = None, **format_dict_kwargs: Any) -> str:
        if max_depth is not None:
            if max_depth == 0:
                return str(self.total())
            dct = cast(LossDict, self.aggregate(max_depth))
        else:
            dct = self

        fmt = build_fmtstr(precision=3, type="e")
        values = [fmt.format(value.item()) for value in dct.values()]
        return format_dict(OrderedDict(zip(dct.keys(), values)), **format_dict_kwargs)

    def __str__(self) -> str:
        return self.format()


class TensorKey:
    def __init__(self, x: torch.Tensor, precision: int = 4) -> None:
        x = x.detach()
        self._key = (*self._extract_meta(x), *self._calculate_stats(x, precision))

    @staticmethod
    def _extract_meta(x: torch.Tensor) -> Tuple[Hashable, ...]:
        return (x.device, x.dtype, x.size())

    @staticmethod
    def _calculate_stats(x: torch.Tensor, precision: int) -> List[str]:
        stat_fns = (torch.min, torch.max, torch.norm)
        return [f"{stat_fn(x):.{precision}e}" for stat_fn in stat_fns]

    @property
    def key(self) -> Tuple[Hashable, ...]:
        return self._key

    def __eq__(self, other) -> bool:
        if isinstance(other, torch.Tensor):
            other = TensorKey(other)
        elif not isinstance(other, TensorKey):
            # FIXME
            raise TypeError

        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return str(self.key)
