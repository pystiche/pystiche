import contextlib
import itertools
import warnings
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
    Type,
    TypeVar,
    Union,
)
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch
from torchvision.datasets.utils import check_md5

__all__ = [
    "prod",
    "to_1d_arg",
    "to_2d_arg",
    "to_3d_arg",
    "zip_equal",
    "build_fmtstr",
    "verify_str_arg",
    "build_complex_obj_repr",
    "get_input_image",
    "build_deprecation_message",
    "get_device",
    "download_file",
    "reduce",
    "suppress_warnings",
]


def prod(iterable: Iterable) -> Any:
    return _reduce(mul, iterable)


T = TypeVar("T")


def _to_nd_arg(dims: int) -> Callable[[Union[T, Sequence[T]]], Tuple[T, ...]]:
    def to_nd_arg(x: Union[T, Sequence[T]]) -> Tuple[T, ...]:
        if isinstance(x, Sequence):
            if len(x) != dims:
                msg = f"Expected sequence of length {dims}, but got {len(x)}."
                raise RuntimeError(msg)
            return tuple(x)
        else:
            return tuple(itertools.repeat(x, dims))

    return to_nd_arg


to_1d_arg = _to_nd_arg(1)
to_2d_arg = _to_nd_arg(2)
to_3d_arg = _to_nd_arg(3)


def zip_equal(*sequences: Sequence) -> Iterable:
    numel = len(sequences[0])
    if any(len(sequence) != numel for sequence in sequences[1:]):
        raise RuntimeError("All sequences should have the same length")
    return zip(*sequences)


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
    else:  # starting_point == "random":
        if content_image is not None:
            return torch.rand_like(content_image)
        elif style_image is not None:
            return torch.rand_like(style_image)
        raise RuntimeError("starting_point is 'random', but no image is given")


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
    url: str,
    file: Optional[str] = None,
    user_agent: Optional[str] = None,
    md5: Optional[str] = None,
) -> str:
    if file is None:
        file = path.basename(url)
    if user_agent is None:
        user_agent = "pystiche"
    else:
        warnings.warn(build_deprecation_message("The parameter user_agent", "0.6.0"))

    request = Request(url, headers={"User-Agent": user_agent})

    try:
        with urlopen(request) as response, open(file, "wb") as fh:
            fh.write(response.read())
    except HTTPError as error:
        msg = f"The server returned {error.code}: {error.reason}."
        raise RuntimeError(msg) from error

    if md5 is not None and not check_md5(file, md5):
        raise RuntimeError(f"The MD5 checksum of {file} mismatches.")

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


@contextlib.contextmanager
def suppress_warnings(*categories: Type[Warning]) -> Iterator[None]:
    if not categories:
        categories = (UserWarning,)
    old_filters = set(warnings.filters)  # type: ignore[attr-defined]
    for category in categories:
        warnings.filterwarnings("ignore", category=category)
    new_filters = set(warnings.filters) - old_filters  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for filter in new_filters:
            warnings.filters.remove(filter)  # type: ignore[attr-defined]
        warnings._filters_mutated()  # type: ignore[attr-defined]
