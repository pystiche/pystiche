from abc import ABC, abstractmethod
from typing import TypeVar, Any, Union, Optional, Callable, Generator, Iterable
from collections import OrderedDict, namedtuple as builtin_namedtuple
import itertools
import torch
from torch import nn
from pystiche.misc import is_valid_variable_name

BuiltinObject = object
BuiltinList = list
BuiltinTuple = tuple
BuiltinSet = set

__all__ = ["list", "tuple", "set", "namedtuple", "object"]

T = TypeVar("T")


class _SpecialObjectBase(ABC):
    @abstractmethod
    def children(self: T) -> Generator[Union[torch.Tensor, nn.Module, T], None, None]:
        yield
        return

    def apply_(self, fn: Callable):
        for obj in self.children():
            if isinstance(obj, torch.Tensor):
                obj.data = fn(obj.data)
                if obj._grad is not None:
                    obj._grad.data = fn(obj._grad.data)
            elif isinstance(obj, (nn.Module, _SpecialObjectBase)):
                obj.apply(fn)

    def apply(self: T, fn: Callable) -> T:
        self.apply_(fn)
        return self

    def cuda(self: T, device: Optional[torch.device] = None) -> T:
        return self.apply(lambda x: x.cuda(device))

    def cpu(self: T) -> T:
        return self.apply(lambda x: x.cpu())

    def type(self: T, dst_type: Union[torch.dtype, str]) -> T:
        return self.apply(lambda x: x.type(dst_type))

    def float(self: T) -> T:
        return self.apply(lambda x: x.float())

    def double(self: T) -> T:
        return self.apply(lambda x: x.double())

    def half(self: T) -> T:
        return self.apply(lambda x: x.half())

    def to(self: T, *args: Any, **kwargs: Any) -> T:
        return self.apply(lambda x: x.to(*args, **kwargs))

    def detach(self: T) -> T:
        return self.apply(lambda x: x.detach())


class _SpecialIterable(_SpecialObjectBase):
    def children(self):
        return iter(self)


class List(_SpecialIterable, BuiltinList):
    pass


def list(*args, **kwargs):
    return List(*args, **kwargs)


class Tuple(_SpecialIterable, BuiltinTuple):
    pass


def tuple(*args, **kwargs):
    return Tuple(*args, **kwargs)


class Set(_SpecialIterable, BuiltinSet):
    pass


def set(*args, **kwargs):
    return Set(*args, **kwargs)


def namedtuple(*args, **kwargs):
    class Namedtuple(_SpecialIterable, builtin_namedtuple(*args, **kwargs)):
        pass

    return Namedtuple


class _SpecialObjectStorage(OrderedDict):
    pass


class object(_SpecialObjectBase):
    _SPECIAL_OBJ_TYPES = (torch.Tensor, nn.Module, _SpecialObjectBase)
    _SPECIAL_OBJ_TYPE_NAMES = (
        "torch.Tensor",
        "torch.nn.Module",
        "pystiche.object",
        "pystiche.list",
        "pystiche.tuple",
        "pystiche.set",
        "pystiche.namedtuple",
    )
    _STR_INDENT = 4

    def __init__(self) -> None:
        self._special_obj_storage = _SpecialObjectStorage()

    def _is_special_obj(self, obj: Any) -> bool:
        return isinstance(obj, self._SPECIAL_OBJ_TYPES)

    def __setattr__(self, name: str, value: Any):
        if name == "_special_obj_storage":
            if not isinstance(value, _SpecialObjectStorage):
                # TODO: add error message
                raise RuntimeError
            BuiltinObject.__setattr__(self, name, value)
            return

        is_initialized = "_special_obj_storage" in self.__dict__
        is_special_obj = self._is_special_obj(value)
        if is_initialized:
            if is_special_obj:
                self._special_obj_storage[name] = value
                if name in self.__dict__:
                    del self.__dict__[name]
            else:
                BuiltinObject.__setattr__(self, name, value)
                if name in self._special_obj_storage:
                    del self._special_obj_storage[name]
        else:
            if is_special_obj:
                msg = (
                    "Cannot assign an object of the following types or derived "
                    "subtypes before __init__() call"
                )
                msg = "\n".join((msg, "", *self._SPECIAL_OBJ_TYPE_NAMES))
                raise AttributeError(msg)
            else:
                BuiltinObject.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._special_obj_storage[name]
        except KeyError:
            msg = "'{}' object has no attribute '{}'".format(type(self).__name__, name)
            raise AttributeError(msg)

    def __delattr__(self, name: str):
        try:
            del self._special_obj_storage[name]
        except KeyError:
            BuiltinObject.__delattr__(self, name)

    def __dir__(self) -> Iterable[str]:
        keys = itertools.chain(
            dir(self.__class__), self.__dict__.keys(), self._special_obj_storage.keys()
        )
        return sorted(BuiltinList(filter(is_valid_variable_name, keys)))

    def children(self):
        return self._special_obj_storage.values()

    def __str__(self) -> str:
        head = self.__class__.__name__ + "("
        body = self.extra_str().splitlines()
        tail = ")"

        if not body:
            return head + tail
        elif len(body) == 1:
            return head + body[0] + tail
        else:
            body = [" " * self._STR_INDENT + row for row in body]
            return "\n".join([head] + body + [tail])

    def extra_str(self) -> str:
        return ""
