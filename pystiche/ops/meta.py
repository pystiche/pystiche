from abc import ABC
from collections import defaultdict
from typing import Union

__all__ = [
    "OperatorCls",
    "UndefinedOperatorCls",
    "Unary",
    "Binary",
    "cls",
    "OperatorDomain",
    "UndefinedOperatorDomain",
    "Pixel",
    "Latent",
    "domain",
]


# TODO: this should be immutable
class OperatorCls(ABC):
    pass


class UndefinedOperatorCls(OperatorCls):
    pass


class Unary(OperatorCls):
    pass


class Binary(OperatorCls):
    pass


TYPE_MAP = defaultdict(UndefinedOperatorCls, {"unary": Unary(), "binary": Binary()})


def cls(cls: Union[str, OperatorCls]) -> OperatorCls:
    if isinstance(cls, OperatorCls):
        return cls
    return TYPE_MAP[cls]


# TODO: this should be immutable
class OperatorDomain(ABC):
    pass


class UndefinedOperatorDomain(OperatorDomain):
    pass


class Pixel(OperatorDomain):
    pass


class Latent(OperatorDomain):
    pass


DOMAIN_MAP = defaultdict(UndefinedOperatorDomain, {"pixel": Pixel(), "latet": Latent()})


def domain(domain: Union[str, OperatorDomain]) -> OperatorDomain:
    if isinstance(domain, OperatorDomain):
        return domain
    return DOMAIN_MAP[domain]
