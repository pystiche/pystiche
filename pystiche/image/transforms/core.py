import itertools
from typing import Iterable, Union

import torch

import pystiche

__all__ = ["Transform", "ComposedTransform"]


class Transform(pystiche.Module):
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
            return [
                transform
                for transform in transform.children()
                if isinstance(transform, Transform)
            ]
        elif isinstance(transform, Transform):
            return (transform,)
        else:
            msg = (  # type: ignore[unreachable]
                f"transforms can either be pystiche.image.transforms.Transform or "
                f"pystiche.image.transforms.ComposedTransform, but got "
                f"{type(transform)}."
            )
            raise TypeError(msg)

    return ComposedTransform(*itertools.chain(*map(unroll, transforms)))
