import contextlib
import hashlib
import itertools
import random
import shutil
import tempfile
import warnings
from collections import OrderedDict
from functools import reduce as _reduce
from operator import mul
from os import path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import requests

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
    "build_complex_obj_repr",
    "is_almost",
    "make_reproducible",
    "get_input_image",
    "get_tmp_dir",
    "get_sha256_hash",
    "save_state_dict",
    "build_deprecation_message",
    "warn_deprecation",
    "get_device",
    "download_file",
    "reduce",
]


def prod(iterable: Iterable) -> Any:
    return _reduce(mul, iterable)


T = TypeVar("T")


def _to_nd_arg(dims: int) -> Callable[[Union[T, Sequence[T]]], Tuple[T, ...]]:
    def to_nd_arg(x: Union[T, Sequence[T]]) -> Tuple[T, ...]:
        if x is None:
            msg = build_deprecation_message(  # type: ignore[unreachable]
                "Passing None as argument",
                "0.4.0",
                info="If you need this behavior, please implement it in the caller.",
            )
            warnings.warn(msg)
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


# TODO: has this function any purpose?
def to_tuplestr(sequence: Sequence) -> str:
    sequence = [str(item) for item in sequence]
    if len(sequence) == 0:
        values = ""
    elif len(sequence) == 1:
        values = sequence[0] + ","
    else:
        values = ", ".join(sequence)
    return f"({values})"


def to_engtuplestr(sequence: Sequence, **kwargs: Any) -> str:
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
) -> str:
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


def format_dict(
    dct: Dict[str, Any], sep: str = ": ", key_align: str = "<", value_align: str = "<"
) -> str:
    msg = build_deprecation_message("The function format_dict", "0.4.0")
    warnings.warn(msg)
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
    arg: Any, param: Optional[str] = None, valid_args: Optional[Sequence[str]] = None
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


def build_complex_obj_repr(
    name: str,
    properties: Optional[Dict[str, Any]] = None,
    named_children: Sequence[Tuple[str, Any]] = (),
    line_length: int = 80,
    num_indent: int = 2,
) -> str:
    def format_properties(properties: Dict[str, Any], sep: str) -> str:
        return sep.join([f"{key}={value}" for key, value in properties.items()])

    def indent(line: str) -> str:
        return " " * num_indent + line

    if properties is None:
        properties = {}

    prefix = f"{name}("
    postfix = ")"

    body = format_properties(properties, ", ")

    body_too_long = (
        len(body) + (num_indent if named_children else len(prefix) + len(postfix))
        > line_length
    )
    multiline_body = len(str(body).splitlines()) > 1

    if body_too_long or multiline_body:
        body = format_properties(properties, ",\n")
    elif not named_children:
        return prefix + body + postfix

    body = [indent(line) for line in body.splitlines()]

    for name, module in named_children:
        lines = str(module).splitlines()
        body.append(indent(f"({name}): {lines[0]}"))
        for line in lines[1:]:
            body.append(indent(line))

    return "\n".join([prefix] + body + [postfix])


def build_obj_str(
    name: str,
    properties: Optional[Dict[str, Any]] = None,
    properties_threshold: Optional[int] = None,
    **kwargs: Any,
) -> str:
    msg = build_deprecation_message(
        "The function build_obj_str",
        "0.4.0",
        info="It was renamed to build_complex_obj_repr.",
    )
    warnings.warn(msg)

    if properties is not None and properties_threshold is not None:
        msg = build_deprecation_message(
            "The parameter properties_threshold",
            "0.4.0",
            info="The line breaks are now controlled by the line_length parameter.",
        )
        warnings.warn(msg)
        line_length = 0 if len(properties) > properties_threshold else 10_000
    else:
        line_length = 80

    return build_complex_obj_repr(name, line_length=line_length, **kwargs)


def is_almost(actual: float, desired: float, eps: float = 1e-6) -> bool:
    msg = build_deprecation_message("The function is_almost", "0.4.0")
    warnings.warn(msg)
    return abs(actual - desired) < eps


def make_reproducible(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        # Both attributes are dynamically assigned to the module. See
        # https://github.com/pytorch/pytorch/blob/a1eaaea288cf51abcd69eb9b0993b1aa9c0ce41f/torch/backends/cudnn/__init__.py#L115-L129
        # The type errors are ignored, since this is still the recommended practice.
        # https://pytorch.org/docs/stable/notes/randomness.html#cudnn
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


def get_input_image(
    starting_point: Union[str, torch.Tensor] = "content",
    content_image: Optional[torch.Tensor] = None,
    style_image: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generates an input image for NST from the given ``starting_point``.

    Args:
        starting_point: If :class:`~torch.Tensor` returns a copy. If ``"content"`` or
         ``"style"`` returns a copy of ``content_image`` or ``style_image``,
         respectively. If ``"random"`` returns a white noise image with the dimensions
         of ``content_image`` or ``style_image``, respectively. Defaults to
         ``"content"``.
        content_image: Content image. Only required if ``starting_point`` is
            ``"content"`` or ``"random"``.
        style_image: Style image. Only required if ``starting_point`` is
            ``"style"`` or ``"random"``.
    """
    if isinstance(starting_point, torch.Tensor):
        return starting_point.clone()

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
    ext: str = ".pth",
    to_cpu: bool = True,
    hash_len: int = 8,
) -> str:
    if isinstance(input, nn.Module):
        state_dict = input.state_dict()
    else:
        state_dict = OrderedDict(input)

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


def warn_deprecation(
    msg_or_description: str,
    version: Optional[str] = None,
    info: Optional[str] = None,
    url: Optional[str] = None,
) -> None:
    msg = build_deprecation_message(
        "META: The function warn_deprecation",
        "0.4.0",
        url="https://github.com/pmeier/pystiche/pull/189",
    )
    warnings.warn(msg, DeprecationWarning)

    if version is not None:
        description = msg_or_description
        msg = build_deprecation_message(description, version, info=info, url=url)
    else:
        msg = msg_or_description
    warnings.warn(msg)


def get_device(device: Optional[str] = None) -> torch.device:
    """Selects a device to perform an NST on.

    Args:
        device: If ``str``, returns the corresponding :class:`~torch.device`. If
            ``None`` selects CUDA if available and otherwise CPU. Defaults to ``None``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def download_file(
    url: str, file: Optional[str] = None, user_agent: str = "pystiche"
) -> str:
    if file is None:
        file = path.basename(url)
    headers = {"User-Agent": user_agent}
    with open(file, "wb") as fh:
        fh.write(requests.get(url, headers=headers).content)
    return file


def reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduces a :class:`~torch.Tensor` as specified.

    Args:
        x: Input tensor.
        reduction: Reduction method to be applied to ``x``. If ``"none"``, no reduction
            will be applied. If ``"sum"`` or ``"mean"``, the :func:`~torch.sum` or
            :func:`~torch.mean` will be applied across all dimensions of ``x``.
    """
    verify_str_arg(reduction, "reduction", ("mean", "sum", "none"))
    if reduction == "mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:  # reduction == "none":
        return x
