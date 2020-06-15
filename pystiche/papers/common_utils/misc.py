import contextlib
from collections import Sequence
from typing import Callable, ContextManager, Optional
from typing import Sequence as SequenceType
from typing import Tuple, TypeVar, Union, overload

import torch
from torch.utils.data.dataloader import DataLoader

from pystiche.image import extract_batch_size, is_single_image, make_batched_image
from pystiche.optim import OptimLogger

__all__ = [
    "same_size_padding",
    "same_size_output_padding",
    "is_valid_padding",
    "batch_up_image",
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
        return tuple(fn(input) for input in inputs)
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


def is_valid_padding(padding: Union[int, SequenceType[int]]) -> bool:
    def is_valid(x):
        return x > 0

    if isinstance(padding, int):
        return is_valid(padding)
    else:
        return all(elementwise(is_valid, padding))


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
