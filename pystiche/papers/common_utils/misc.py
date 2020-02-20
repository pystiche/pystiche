from typing import (
    Union,
    Optional,
    Sequence as SequenceType,
    Tuple,
    Dict,
    Callable,
    TypeVar,
    ContextManager,
    overload,
)
from collections import Sequence, OrderedDict
import contextlib
from os import path
import shutil
import tempfile
import hashlib
import torch
from torch import nn
from torch.hub import _get_torch_home
from torch.utils.data.dataloader import DataLoader
from pystiche.image import is_single_image, make_batched_image, extract_batch_size
from pystiche.optim import OptimLogger

__all__ = [
    "same_size_padding",
    "same_size_output_padding",
    "batch_up_image",
    "get_tmp_dir",
    "get_sha256_hash",
    "save_state_dict",
    "paper_replication",
]

In = TypeVar("In")
Out = TypeVar("Out")


@overload
def elementwise(fn: Callable[[In], Out], inputs: In) -> Out:
    ...


@overload
def elementwise(fn: Callable[[In], Out], input: SequenceType[In]) -> Tuple[Out, ...]:
    ...


def elementwise(
    fn: Callable[[In], Out], inputs: Union[In, SequenceType[In]]
) -> Union[Out, Tuple[Out, ...]]:
    if isinstance(inputs, Sequence):
        return tuple([fn(input) for input in inputs])
    return fn(inputs)


@overload
def same_size_padding(kernel_size: int) -> int:
    ...


@overload
def same_size_padding(kernel_size: SequenceType[int]) -> Tuple[int, ...]:
    ...


def same_size_padding(
    kernel_size: Union[int, SequenceType[int]]
) -> Union[int, Tuple[int, ...]]:
    return elementwise(lambda x: (x - 1) // 2, kernel_size)


@overload
def same_size_output_padding(stride: int) -> int:
    ...


@overload
def same_size_output_padding(stride: SequenceType[int]) -> Tuple[int, ...]:
    ...


def same_size_output_padding(stride: Union[int, SequenceType[int]]) -> Tuple[int, ...]:
    return elementwise(lambda x: x - 1, stride)


def batch_up_image(
    image: torch.Tensor,
    desired_batch_size: Optional[int] = None,
    loader: Optional[DataLoader] = None,
) -> torch.Tensor:
    if desired_batch_size is None and loader is None:
        raise RuntimeError

    if is_single_image(image):
        image = make_batched_image(image)
    elif extract_batch_size(image) > 1:
        raise RuntimeError

    if desired_batch_size is None:
        desired_batch_size = loader.batch_size
        if desired_batch_size is None:
            try:
                desired_batch_size = loader.batch_sampler.batch_size
            except AttributeError:
                raise RuntimeError

    return image.repeat(desired_batch_size, 1, 1, 1)


# FIXME: move to pystiche.misc
@contextlib.contextmanager
def get_tmp_dir(**mkdtemp_kwargs) -> ContextManager[str]:
    tmp_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


# FIXME: move to pystiche.misc
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


@contextlib.contextmanager
def paper_replication(
    optim_logger: OptimLogger, title: str, url: str, author: str, year: Union[str, int]
) -> ContextManager:
    header = "\n".join(
        (
            "Replication of the paper",
            f"'{title}'",
            url,
            "authored by",
            author,
            f"in {str(year)}",
        )
    )
    with optim_logger.environment(header):
        yield
