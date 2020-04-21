import contextlib
import hashlib
import itertools
import random
import shutil
import tempfile
import warnings
from collections import OrderedDict
from functools import reduce
from operator import mul
from os import path
from typing import (
    Any,
    ContextManager,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
)

import numpy as np

import torch
from torch import nn
from torch.hub import _get_torch_home

__all__ = [
    "prod",
    "to_1d_arg",
    "to_2d_arg",
    "to_3d_arg",
    "zip_equal",
    "to_eng",
    "to_engstr",
    "to_tuplestr",
    "to_engtuplestr",
    "build_fmtstr",
    "format_dict",
    "verify_str_arg",
    "build_obj_str",
    "is_almost",
    "make_reproducible",
    "get_input_image",
    "get_tmp_dir",
    "get_sha256_hash",
    "save_state_dict",
    "build_deprecation_message",
    "warn_deprecation",
]


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


# TODO: has this function any purpose?git
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
#  see https://pyformat.info/#param_align
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
        properties = ",\n".join([f"{key}={value}" for key, value in properties.items()])

    def indent(line):
        return " " * num_indent + line

    body = [indent(line) for line in properties.splitlines()]

    for name, module in named_children:
        lines = str(module).splitlines()
        body.append(indent(f"({name}): {lines[0]}"))
        for line in lines[1:]:
            body.append(indent(line))

    return "\n".join([prefix] + body + [postfix])


def is_almost(actual: float, desired: float, eps=1e-6):
    return abs(actual - desired) < eps


def make_reproducible(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_input_image(
    starting_point: Union[str, torch.Tensor] = "content",
    content_image: Optional[torch.tensor] = None,
    style_image: Optional[torch.tensor] = None,
):
    if isinstance(starting_point, torch.Tensor):
        return starting_point

    starting_point = verify_str_arg(
        starting_point, "starting_point", ("content", "style", "random")
    )

    if starting_point == "content":
        if content_image is not None:
            return content_image.clone()
        raise RuntimeError("starting_point is 'content', but no content image is given")
    elif starting_point == "style":
        if style_image is not None:
            return style_image.clone()
        raise RuntimeError("starting_point is 'style', but no style image is given")
    elif starting_point == "random":
        if content_image is not None:
            return torch.rand_like(content_image)
        elif style_image is not None:
            return torch.rand_like(style_image)
        raise RuntimeError("starting_point is 'random', but no image is given")

    raise RuntimeError


@contextlib.contextmanager
def get_tmp_dir(**mkdtemp_kwargs) -> ContextManager[str]:
    tmp_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


def get_sha256_hash(file: str, chunk_size: int = 4096) -> str:
    hasher = hashlib.sha256()
    with open(file, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_state_dict(
    input: Union[Dict[str, torch.Tensor], nn.Module()],
    name: str,
    root: Optional[str] = None,
    ext=".pth",
    to_cpu: bool = True,
    hash_len: int = 8,
) -> str:
    if isinstance(input, nn.Module):
        state_dict = input.state_dict()
    else:
        state_dict = input

    if to_cpu:
        state_dict = OrderedDict(
            [(key, tensor.detach().cpu()) for key, tensor in state_dict.items()]
        )

    if root is None:
        root = _get_torch_home()

    with get_tmp_dir() as tmp_dir:
        tmp_file = path.join(tmp_dir, "tmp")
        torch.save(state_dict, tmp_file)
        sha256 = get_sha256_hash(tmp_file)

        file = path.join(root, f"{name}-{sha256[:hash_len]}{ext}")
        shutil.move(tmp_file, file)

    return file


def build_deprecation_message(
    description: str,
    version: str,
    info: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    msg = (
        f"{description.strip()} is deprecated since pystiche=={version} and will be "
        "removed in a future release."
    )
    if info is not None:
        msg += f" {info.strip()}"
    if url is not None:
        msg += f" See {url} for further details."
    return msg


def warn_deprecation(*args: str, **kwargs: Optional[str]):
    msg = build_deprecation_message(
        "META: The function warn_deprecation",
        "0.4.0",
        url="https://github.com/pmeier/pystiche/pull/189",
    )
    warnings.warn(msg, DeprecationWarning)
    if len(args) == 1 and not kwargs:
        msg = args[0]
    else:
        msg = build_deprecation_message(*args, **kwargs)
    warnings.warn(msg, UserWarning)


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(device, str):
        device = torch.device(device)

    return device
