from typing import Union, Optional, Sequence, Collection, Iterator
import itertools
import numpy as np
from pystiche import Object
from pystiche.misc import zip_equal
from pystiche.ops import (
    Operator,
    ComparisonOperator,
    EncodingOperator,
    Guidance,
    ComparisonGuidance,
)
from .level import PyramidLevel
from .storage import ImageStorage


__all__ = ["ImagePyramid", "OctaveImagePyramid"]


class ImagePyramid(Object):
    def __init__(
        self,
        edge_sizes: Sequence[int],
        num_steps: Union[Sequence[int], int],
        edge: Union[Sequence[str], str] = "short",
        interpolation_mode: str = "bilinear",
        resize_targets: Collection[Operator] = (),
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
            except Exception as exc:
                raise exc
            finally:
                image_storage.restore()

    def _resize(self, level: PyramidLevel):
        for op in self._resize_ops():
            if isinstance(op, ComparisonGuidance) and op.has_target_guide:
                resized_guide = level.resize_guide(op.target_guide)
                op.set_target_guide(resized_guide, recalc_repr=False)

            if isinstance(op, ComparisonOperator) and op.has_target_image:
                resized_image = level.resize_image(
                    op.target_image, interpolation_mode=self.interpolation_mode
                )
                op.set_target_image(resized_image)

            if isinstance(op, Guidance) and op.has_input_guide:
                resized_guide = level.resize_guide(op.input_guide)
                op.set_input_guide(resized_guide)

    def _resize_ops(self) -> Iterator[Operator]:
        unique_modules = set(
            itertools.chain(*[target.modules() for target in self._resize_targets])
        )
        for module in unique_modules:
            if isinstance(module, Operator):
                yield module

    def _properties(self):
        dct = super()._properties()
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        return dct

    def _named_children(self):
        yield from super()._named_children()
        yield from enumerate(self._levels)


class OctaveImagePyramid(ImagePyramid):
    def __init__(
        self,
        max_edge_size: int,
        num_steps: Union[int, Sequence[int]],
        num_levels: Optional[int] = None,
        min_edge_size: int = 64,
        **kwargs,
    ):
        if num_levels is None:
            num_levels = int(np.floor(np.log2(max_edge_size / min_edge_size))) + 1

        edge_sizes = [
            round(max_edge_size / (2.0 ** ((num_levels - 1) - level)))
            for level in range(num_levels)
        ]
        super().__init__(edge_sizes, num_steps, **kwargs)
