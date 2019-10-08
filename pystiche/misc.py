from typing import overload, Any, Optional, Iterator, Iterable, Sequence, Sized, Tuple
from keyword import iskeyword
from functools import reduce
import itertools
from operator import mul
import numpy as np


def prod(iterable: Iterable) -> Any:
    return reduce(mul, iterable)


def _to_nd_arg(x: Any, dims: int) -> Any:
    if x is None:
        return None

    if isinstance(x, Sized):
        assert len(x) == dims
        y = x
    else:
        y = itertools.repeat(x, dims)
    return tuple(y)


def to_1d_arg(x: Any) -> Any:
    return _to_nd_arg(x, 1)


def to_2d_arg(x: Any) -> Any:
    return _to_nd_arg(x, 2)


def to_3d_arg(x: Any) -> Any:
    return _to_nd_arg(x, 3)


def zip_equal(*sequences: Sequence) -> Iterable:
    numel = len(sequences[0])
    if not all([len(sequence) == numel for sequence in sequences[1:]]):
        raise RuntimeError("All sequences should have the same length")
    return zip(*sequences)


def to_eng(num: float, eps: float = 1e-8) -> Tuple[float, int]:
    if np.abs(num) < eps:
        return 0.0, 0

    exp = np.floor(np.log10(np.abs(num))).astype(np.int)
    exp -= np.mod(exp, 3)
    sig = num * 10.0 ** -exp

    return sig, exp


def to_engstr(
    num: float, digits: int = 4, exp_sep: str = "e", eps: float = 1e-8
) -> str:
    sig, exp = to_eng(num, eps=eps)
    mag = np.abs(sig)

    if mag < 1.0 - eps:
        return "0"

    fmt_str = "{:." + str(digits) + "g}"

    if exp == -3 and mag > 1.0 + eps:
        return fmt_str.format(num)

    sigstr = fmt_str.format(sig)
    expstr = (exp_sep + str(exp)) if exp != 0 else ""
    return sigstr + expstr


def to_tuplestr(sequence: Sequence) -> str:
    sequence = [str(item) for item in sequence]
    if len(sequence) == 0:
        values = ""
    elif len(sequence) == 1:
        values = sequence[0] + ","
    else:
        values = ", ".join(sequence)
    return "({})".format(values)


def to_engtuplestr(sequence: Sequence, **kwargs) -> str:
    return to_tuplestr([to_engstr(item, **kwargs) for item in sequence])


def maxlen_fmtstr(iterable, identifier="", type="s", precision=""):
    width = str(max(map(len, iterable)))
    if precision:
        precision = str(precision)
        if not precision.startswith("."):
            precision = "." + precision
    return "{" + identifier + ":" + width + precision + type + "}"


def verify_str_arg(
    arg: str, param: str = None, valid_args: Sequence[str] = None
) -> str:
    if not isinstance(arg, str):
        if param is None:
            msg1 = "Expected type str"
        else:
            msg1 = "Expected type str for parameter {}".format(param)
        msg2 = ", but got type {}.".format(type(arg))
        raise ValueError(msg1 + msg2)

    if valid_args is None:
        return arg

    if arg not in valid_args:
        if param is None:
            msg1 = "Unknown argument '{}'. ".format(arg)
        else:
            msg1 = "Unknown argument '{}' for parameter {}. ".format(arg, param)
        msg2 = "Valid arguments are {{{}}}."
        msg2 = msg2.format("'" + "', '".join(valid_args) + "'")
        raise ValueError(msg1 + msg2)

    return arg


def is_valid_variable_name(name: str) -> bool:
    name = str(name)
    return name.isidentifier() and not iskeyword(name)


def subclass_iterator(
    sequence: Sequence,
    *subclasses: Any,
    not_instance: bool = False,
    all_subclasses: bool = True
) -> Iterator[Any]:
    if not subclasses:
        return iter(sequence)

    mask = [[isinstance(obj, subclass) for subclass in subclasses] for obj in sequence]
    reduce_fn = all if all_subclasses else any
    mask = map(lambda x: not_instance ^ reduce_fn(x), mask)
    return itertools.compress(sequence, mask)
