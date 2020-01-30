from typing import (
    Any,
    Union,
    Optional,
    Iterator,
    Iterable,
    Sequence,
    Sized,
    Tuple,
    Dict,
)
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
    return f"({values})"


def to_engtuplestr(sequence: Sequence, **kwargs) -> str:
    return to_tuplestr([to_engstr(item, **kwargs) for item in sequence])


# FIXME: add padding
# FIXME: add sign
def build_fmtstr(
    id: Optional[Union[int, str]] = None,
    align: Optional[str] = None,
    field_len: Optional[Union[int, str]] = None,
    precision: Optional[Union[int, str]] = None,
    type: Optional[str] = None,
):
    fmtstr = r"{"
    if id is not None:
        fmtstr += str(id)
    fmtstr += ":"
    if align is not None:
        fmtstr += align
    if field_len is not None:
        fmtstr += str(field_len)
    if precision is not None:
        fmtstr += "." + str(precision)
    if type is not None:
        fmtstr += type
    fmtstr += r"}"
    return fmtstr


# FIXME: this should be able to handle multi line values
def format_dict(dct: Dict[str, Any], sep=": ", key_align="<", value_align="<"):
    key_field_len, val_field_len = [
        max(lens)
        for lens in zip(*[(len(key), len(str(val))) for key, val in dct.items()])
    ]

    fmtstr = build_fmtstr(id=0, align=key_align, field_len=key_field_len, type="s")
    fmtstr += sep
    fmtstr += build_fmtstr(id=1, align=value_align, field_len=val_field_len, type="s")

    lines = [fmtstr.format(key, str(val)) for key, val in dct.items()]
    return "\n".join(lines)


def verify_str_arg(
    arg: str, param: str = None, valid_args: Sequence[str] = None
) -> str:
    if not isinstance(arg, str):
        if param is None:
            msg1 = "Expected type str"
        else:
            msg1 = f"Expected type str for parameter {param}"
        msg2 = f", but got type {type(arg)}."
        raise ValueError(msg1 + msg2)

    if valid_args is None:
        return arg

    if arg not in valid_args:
        if param is None:
            msg1 = f"Unknown argument '{arg}'. "
        else:
            msg1 = f"Unknown argument '{arg}' for parameter {param}. "
        msg2 = "Valid arguments are {{{}}}."
        msg2 = msg2.format("'" + "', '".join(valid_args) + "'")
        raise ValueError(msg1 + msg2)

    return arg


def build_obj_str(
    name: str,
    properties: Dict[str, Any] = None,
    named_children: Sequence[Tuple[str, Any]] = (),
    properties_threshold: int = 4,
    num_indent: int = 2,
):
    prefix = f"{name}("
    postfix = ")"

    if properties is None:
        properties = {}

    num_properties = len(properties)
    multiline_properties = any(
        [len(str(value).splitlines()) > 1 for value in properties.values()]
    )
    if not multiline_properties and num_properties < properties_threshold:
        properties = ", ".join([f"{key}={value}" for key, value in properties.items()])

        if not named_children:
            return prefix + properties + postfix
    else:
        properties = format_dict(properties, sep="=")

    def indent(line):
        return " " * num_indent + line

    body = [indent(line) for line in properties.splitlines()]

    for name, module in named_children:
        lines = str(module).splitlines()
        body.append(indent(f"({name}): {lines[0]}"))
        for line in lines[1:]:
            body.append(indent(line))

    return "\n".join([prefix] + body + [postfix])
