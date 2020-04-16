from abc import ABC
from collections import defaultdict
from typing import Optional, Union

__all__ = [
    "OperatorCls",
    "UndefinedOperatorCls",
    "Regularization",
    "Comparison",
    "cls",
    "OperatorDomain",
    "UndefinedOperatorDomain",
    "Pixel",
    "Encoding",
    "domain",
]


# TODO: this should be immutable
class OperatorCls(ABC):
    pass


class UndefinedOperatorCls(OperatorCls):
    pass


class Regularization(OperatorCls):
    pass


class Comparison(OperatorCls):
    pass


TYPE_MAP = defaultdict(
    UndefinedOperatorCls,
    {"regularization": Regularization(), "comparison": Comparison()},
)


def cls(name_or_cls: Optional[Union[str, OperatorCls]]) -> OperatorCls:
    if isinstance(name_or_cls, OperatorCls):
        return name_or_cls
    return TYPE_MAP[name_or_cls]


# TODO: this should be immutable
class OperatorDomain(ABC):
    pass


class UndefinedOperatorDomain(OperatorDomain):
    pass


class Pixel(OperatorDomain):
    pass


class Encoding(OperatorDomain):
    pass


DOMAIN_MAP = defaultdict(
    UndefinedOperatorDomain, {"pixel": Pixel(), "encoding": Encoding()}
)


def domain(name_or_domain: Optional[Union[str, OperatorDomain]]) -> OperatorDomain:
    if isinstance(name_or_domain, OperatorDomain):
        return name_or_domain
    return DOMAIN_MAP[name_or_domain]