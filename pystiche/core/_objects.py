import warnings
from abc import ABC
from collections import OrderedDict
from copy import copy
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Union,
    cast,
)

import torch

from pystiche.meta import is_scalar_tensor
from pystiche.misc import (
    build_complex_obj_repr,
    build_deprecation_message,
    build_fmtstr,
    format_dict,
)

__all__ = ["ComplexObject", "LossDict", "TensorKey"]


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
        return iter(())

    def extra_named_children(self) -> Iterator[Tuple[str, Any]]:
        return iter(())

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
    def __init__(self) -> None:
        msg = build_deprecation_message(
            "The class Object", "0.4", info="It was renamed to ComplexObject."
        )
        warnings.warn(msg)


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
                msg = "loss is a torch.Tensor but is not scalar."
                raise TypeError(msg)
            super().__setitem__(name, loss)
        elif isinstance(loss, LossDict):
            for child_name, child_loss in loss.items():
                super().__setitem__(f"{name}.{child_name}", child_loss)
        else:
            msg = (  # type: ignore[unreachable]
                f"loss can be a scalar torch.Tensor or a pystiche.LossDict, but got "
                f"a {type(loss)} instead."
            )
            raise TypeError(msg)

    def aggregate(self, depth: int) -> Union[torch.Tensor, "LossDict"]:
        def sum_partial_losses(partial_losses: Iterable[torch.Tensor]) -> torch.Tensor:
            return cast(torch.Tensor, sum(partial_losses))

        if depth == 0:
            return sum_partial_losses(self.values())

        splits = [name.split(".") for name in self.keys()]
        if not any([len(split) >= depth for split in splits]):
            return copy(self)

        agg_names = [".".join(split[:depth]) for split in splits]
        key_map = dict(zip(self.keys(), agg_names))
        agg_losses: Dict[str, List[torch.Tensor]] = {
            name: [] for name in set(agg_names)
        }
        for name, loss in self.items():
            agg_losses[key_map[name]].append(loss)

        return LossDict(
            [(name, sum_partial_losses(agg_losses[name])) for name in agg_names]
        )

    def total(self) -> torch.Tensor:
        return cast(torch.Tensor, self.aggregate(0))

    def backward(self, *args: Any, **kwargs: Any) -> None:
        self.total().backward(*args, **kwargs)

    def item(self) -> float:
        return self.total().item()

    def __float__(self) -> float:
        return self.item()

    def __mul__(self, other: SupportsFloat) -> "LossDict":
        other = float(other)
        return LossDict([(name, loss * other) for name, loss in self.items()])

    # TODO: can this be moved in __str__?
    def format(self, depth: Optional[int] = None, **format_dict_kwargs: Any) -> str:
        if depth is not None:
            if depth == 0:
                return str(self.total())
            dct = cast(LossDict, self.aggregate(depth))
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

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, torch.Tensor):
            other = TensorKey(other)
        if isinstance(other, TensorKey):
            return self.key == other.key

        # FIXME
        raise TypeError

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return str(self.key)
