from abc import ABC
from collections import OrderedDict
from copy import copy
from typing import (
    Any,
    Callable,
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
from pystiche.misc import build_complex_obj_repr, build_fmtstr

__all__ = ["ComplexObject", "LossDict", "TensorKey"]


class ComplexObject(ABC):
    r"""Object with a complex representation. See
    :func:`pystiche.misc.build_complex_obj_repr` for details.
    """
    _STR_INDENT = 2

    def _properties(self) -> Dict[str, Any]:
        r"""
        Returns:
            Internal properties.

        .. note::

            If subclassed, this method should integrate the new properties in the
            properties of the superclass.
        """
        return OrderedDict()

    def extra_properties(self) -> Dict[str, Any]:
        r"""
        Returns:
            Extra properties.
        """
        return OrderedDict()

    def properties(self) -> Dict[str, Any]:
        r"""
        Returns:
            Internal and extra properties.
        """
        dct = self._properties()
        dct.update(self.extra_properties())
        return dct

    def _named_children(self) -> Iterator[Tuple[str, Any]]:
        r"""
        Yields:
            Internal named children.

        .. note::

            If subclassed, this method should yield the named children of the
            superclass alongside yielding the new named children.
        """
        return iter(())

    def extra_named_children(self) -> Iterator[Tuple[str, Any]]:
        r"""
        Yields:
            Extra named children.
        """
        return iter(())

    def named_children(self) -> Iterator[Tuple[str, Any]]:
        r"""
        Yields:
            Internal and extra named children.
        """
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


class LossDict(OrderedDict):
    r"""Hierarchic dictionary of scalar :class:`torch.Tensor` losses. Levels are
    seperated by ``"."`` in the names.

    Args:
        losses: Optional named losses.
    """

    def __init__(
        self, losses: Sequence[Tuple[str, Union[torch.Tensor, "LossDict"]]] = (),
    ) -> None:
        super().__init__()
        for name, loss in losses:
            self[name] = loss

    def __setitem__(self, name: str, loss: Union[torch.Tensor, "LossDict"]) -> None:
        r"""Add a named loss to the entries.

        Args:
            name: Name of the loss.
            loss: If :class:`torch.Tensor`, it has to be scalar. If :class:`LossDict`,
                it is unpacked and the entries are added as level below of ``name``.

        Raises:
            TypeError: If loss is :class:`torch.Tensor` but isn't scalar.
        """
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

    def aggregate(self, max_depth: int) -> Union[torch.Tensor, "LossDict"]:
        r"""Aggregate all entries up to a given maximum depth.

        Args:
            max_depth: If ``0`` returns sum of all entries as scalar
                :class:`torch.Tensor`.
        """

        def sum_partial_losses(partial_losses: Iterable[torch.Tensor]) -> torch.Tensor:
            return cast(torch.Tensor, sum(partial_losses))

        if max_depth == 0:
            return sum_partial_losses(self.values())

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

        return LossDict(
            [(name, sum_partial_losses(agg_losses[name])) for name in agg_names]
        )

    def total(self) -> torch.Tensor:
        r"""
        Returns:
            Sum of all entries as scalar tensor.
        """
        return cast(torch.Tensor, self.aggregate(0))

    def backward(self, *args: Any, **kwargs: Any) -> None:
        r"""Computes the gradient of all entries with respect to the graph leaves. See
        :meth:`torch.Tensor.backward` for details.
        """
        self.total().backward(*args, **kwargs)

    def item(self) -> float:
        r"""
        Returns:
            The sum of all entries as standard Python number.
        """
        return self.total().item()

    def __float__(self) -> float:
        return self.item()

    def __mul__(self, other: SupportsFloat) -> "LossDict":
        r"""Multiplies all entries with a scalar.

        Args:
            other: Scalar multiplier.
        """
        other = float(other)
        return LossDict([(name, loss * other) for name, loss in self.items()])

    def __str__(self) -> str:
        key_fmtstr = build_fmtstr(
            field_len=max([len(key) for key in self.keys()]), type="s"
        )
        value_fmtstr = build_fmtstr(precision=3, type="e")
        fmtstr = f"{key_fmtstr}: {value_fmtstr}"
        return "\n".join([fmtstr.format(key, value) for key, value in self.items()])


class TensorKey:
    def __init__(self, x: torch.Tensor, precision: int = 4) -> None:
        x = x.detach()
        self._key = (*self._extract_meta(x), *self._calculate_stats(x, precision))

    @staticmethod
    def _extract_meta(x: torch.Tensor) -> Tuple[Hashable, ...]:
        return (x.device, x.dtype, x.size())

    @staticmethod
    def _calculate_stats(x: torch.Tensor, precision: int) -> List[str]:
        stat_fns: Tuple[Callable[[torch.Tensor], torch.Tensor], ...] = (
            torch.min,
            torch.max,
            torch.norm,
        )
        return [f"{stat_fn(x).item():.{precision}e}" for stat_fn in stat_fns]

    @property
    def key(self) -> Tuple[Hashable, ...]:
        return self._key

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, torch.Tensor):
            other = TensorKey(other)

        return self.key == other.key if isinstance(other, TensorKey) else False

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return str(self.key)
