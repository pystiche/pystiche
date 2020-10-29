import itertools
import warnings
from typing import Any, Iterable, Union

import torch

import pystiche
from pystiche.misc import build_deprecation_message

__all__ = ["Transform", "ComposedTransform"]


class Transform(pystiche.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        msg = build_deprecation_message(
            "Using functionality from pystiche.image.transforms",
            "0.7.0",
            info="See https://github.com/pmeier/pystiche/issues/382 for details.",
        )
        warnings.warn(msg, UserWarning)
        super().__init__(*args, **kwargs)

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
