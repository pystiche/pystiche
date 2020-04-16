from abc import ABC
from collections import defaultdict
from typing import Union

__all__ = [
    "Cls",
    "UndefinedCls",
    "Unary",
    "Binary",
    "cls",
    "Domain",
    "UndefinedDomain",
    "Pixel",
    "Latent",
    "domain",
]


# make them immutable
class Cls(ABC):
    pass


class UndefinedCls(Cls):
    pass


class Unary(Cls):
    pass


class Binary(Cls):
    pass


# default_dict
TYPE_MAP = defaultdict(UndefinedCls, {"unary": Unary(), "binary": Binary()})


def cls(cls: Union[str, Cls]) -> Cls:
    if isinstance(cls, Cls):
        return cls
    return TYPE_MAP[cls]


class Domain(ABC):
    pass


class UndefinedDomain(Domain):
    pass


class Pixel(Domain):
    pass


class Latent(Domain):
    pass


DOMAIN_MAP = defaultdict(UndefinedDomain, {"pixel": Pixel(), "latet": Latent()})


def domain(domain: Union[str, Domain]) -> Domain:
    if isinstance(domain, Domain):
        return domain
    return DOMAIN_MAP[domain]
