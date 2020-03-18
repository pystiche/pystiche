from typing import (
    Any,
    Union,
    Optional,
    Iterator,
    Iterable,
    Sequence,
    Tuple,
    Dict,
    Callable,
    TypeVar,
)
import contextlib
from collections import OrderedDict
from functools import reduce
from operator import mul
import warnings
import itertools
from os import path
import shutil
import hashlib
import tempfile
import random
import numpy as np
import torch
from torch.backends import cudnn
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

T = TypeVar("T")


def prod(iterable: Iterable) -> Any:
    return reduce(mul, iterable)


def _to_nd_arg(dims: int) -> Callable[[Union[T, Sequence[T]]], Tuple[T, ...]]:
    def to_nd_arg(x: Union[T, Sequence[T]]) -> Tuple[T, ...]:
        if x is None:
            warn_deprecation(
                "argument",
                "None",
                "0.4",
                info="If you need this behavior, please implement it in the caller.",
            )
            return None

        if isinstance(x, Sequence):
            if len(x) != dims:
                raise RuntimeError
            return tuple(x)
        else:
            return tuple(itertools.repeat(x, dims))

    return to_nd_arg


to_1d_arg = _to_nd_arg(1)
to_2d_arg = _to_nd_arg(2)
to_3d_arg = _to_nd_arg(3)


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
    properties: Optional[Dict[str, Any]] = None,
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

    def join_properties(sep: str) -> str:
        return sep.join([f"{key}={value}" for key, value in properties.items()])

    if not multiline_properties and num_properties < properties_threshold:
        properties = join_properties(", ")

        if not named_children:
            return prefix + properties + postfix
    else:
        properties = join_properties(",\n")

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
    if torch.backends.cudnn.is_available():
        # Both attributes are dynamically assigned. See
        # https://github.com/pytorch/pytorch/blob/a1eaaea288cf51abcd69eb9b0993b1aa9c0ce41f/torch/backends/cudnn/__init__.py#L115-L129
        # The type errors are ignored, since this is still the recommended practice.
        # https://pytorch.org/docs/stable/notes/randomness.html#cudnn
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


def get_input_image(
    starting_point: Union[str, torch.Tensor] = "content",
    content_image: Optional[torch.Tensor] = None,
    style_image: Optional[torch.Tensor] = None,
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
def get_tmp_dir(**mkdtemp_kwargs: Any) -> Iterator[str]:
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
    input: Union[Dict[str, torch.Tensor], nn.Module],
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
    type: str,
    name: str,
    version: str,
    info: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    msg = (
        f"The {type} {name} is deprecated since pystiche=={version} and will be "
        "removed in a future release."
    )
    if info is not None:
        msg += f" {info.strip()}"
    if url is not None:
        msg += f" See {url} for further details."
    return msg


def warn_deprecation(msg_or_type: str, *args: str, **kwargs: Optional[str]):
    if args:
        type = msg_or_type
        name, version = args
        msg = build_deprecation_message(type, name, version, **kwargs)
    else:
        msg = msg_or_type

    warnings.warn(msg, UserWarning)
