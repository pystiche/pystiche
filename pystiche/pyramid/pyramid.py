from typing import Union, Optional, Tuple, Sequence, Collection, Iterator
from collections import OrderedDict
import itertools
import numpy as np
import torch
from torch import nn
import pystiche
from pystiche.misc import zip_equal
from pystiche.ops import (
    Operator,
    ComparisonOperator,
    PixelComparisonOperator,
    EncodingComparisonOperator,
)
from .level import PyramidLevel


__all__ = ["ImagePyramid", "OctaveImagePyramid"]


class ImageStorage:
    def __init__(self, ops):
        self.target_images = {}
        self.input_guides = {}
        self.target_guides = {}
        for op in ops:
            if isinstance(op, (PixelComparisonOperator, EncodingComparisonOperator)):
                self.target_images[op] = op.target_image

    def restore(self):
        for op, target_image in self.target_images.items():
            op.set_target_image(target_image)


class ImagePyramid:
    def __init__(
        self,
        edge_sizes: Sequence[int],
        num_steps: Union[Sequence[int], int],
        edge: Union[Sequence[str], str] = "short",
        interpolation_mode: str = "bilinear",
        resize_targets: Optional[Collection[Operator]] = None,
    ):
        num_levels = len(edge_sizes)
        if isinstance(num_steps, int):
            num_steps = [num_steps] * num_levels
        if isinstance(edge, str):
            edge = [edge] * num_levels

        self._levels = [
            PyramidLevel(edge_size, num_steps_, edge_)
            for edge_size, num_steps_, edge_ in zip_equal(edge_sizes, num_steps, edge)
        ]

        self.interpolation_mode = interpolation_mode
        self._resize_targets = set(resize_targets)

    def add_resize_target(self, op: Operator):
        self._resize_targets.add(op)

    def __len__(self):
        return len(self._levels)

    def __getitem__(self, idx):
        return self._levels[idx]

    def __iter__(self):
        image_storage = ImageStorage(self._resize_ops())
        for level in self._levels:
            try:
                self._resize(level)
                yield level
            finally:
                image_storage.restore()

    def _resize(self, level: PyramidLevel):
        for op in self._resize_ops():
            if isinstance(op, (PixelComparisonOperator, EncodingComparisonOperator)):
                resized_image = level.resize_image(
                    op.target_image, interpolation_mode=self.interpolation_mode
                )
                op.set_target_image(resized_image)

    def _resize_ops(self) -> Iterator[Operator]:
        modules = itertools.chain(
            *[target.modules() for target in self._resize_targets]
        )
        ops = set([op for op in modules if isinstance(op, Operator)])
        for op in ops:
            yield op


class OctaveImagePyramid(ImagePyramid):
    def __init__(
        self,
        max_edge_size: int,
        num_steps: Union[Sequence[int], int],
        num_levels: Optional[int] = None,
        min_edge_size: int = 64,
        **kwargs
    ):
        if num_levels is None:
            num_levels = int(np.floor(np.log2(max_edge_size / min_edge_size))) + 1

        edge_sizes = [
            round(max_edge_size / (2.0 ** ((num_levels - 1) - level)))
            for level in range(num_levels)
        ]
        super().__init__(edge_sizes, num_steps, **kwargs)
