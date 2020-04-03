from typing import Union, Iterable
from abc import abstractmethod
import itertools
import torch
import pystiche

__all__ = ["Transform", "ComposedTransform"]


class Transform(pystiche.Module):
    @abstractmethod
    def forward(self, *input):
        pass

    def __add__(
        self, other: Union["Transform", "ComposedTransform"]
    ) -> "ComposedTransform":
        return compose_transforms(self, other)


class ComposedTransform(Transform):
    def __init__(self, *transforms: Transform):
        super().__init__(indexed_children=transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x

    def __add__(
        self, other: Union["Transform", "ComposedTransform"]
    ) -> "ComposedTransform":
        return compose_transforms(self, other)


def compose_transforms(
    *transforms: Union[Transform, ComposedTransform]
) -> ComposedTransform:
    def unroll(
        transform: Union[Transform, ComposedTransform]
    ) -> Iterable[Union[Transform, ComposedTransform]]:
        if isinstance(transform, ComposedTransform):
            return transform.children()
        elif isinstance(transform, Transform):
            return (transform,)
        else:
            raise TypeError

    return ComposedTransform(*itertools.chain(*map(unroll, transforms)))
