from typing import (
    Any,
    Collection,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from pystiche import ComplexObject
from pystiche.loss import MultiOperatorLoss
from pystiche.misc import zip_equal
from pystiche.ops import ComparisonOperator, Operator, OperatorContainer

from .level import PyramidLevel
from .storage import ImageStorage

__all__ = ["ImagePyramid", "OctaveImagePyramid"]


class ImagePyramid(ComplexObject):
    r"""Image pyramid for a coarse-to-fine optimization on different levels. If
    iterated on yields :class:`~pystiche.pyramid.PyramidLevel` s and handles the
    resizing of all set images and guides of ``resize_targets``.

    Args:
        edge_sizes: Edge sizes for each level.
        num_steps: Number of steps for each level. If sequence of ``int`` its length
            has to match the length of ``edge_sizes``.
        edge: Corresponding edge to the edge size for each level. Can be ``"short"`` or
            ``"long"``. If sequence of ``str`` its length has to match the length of
            ``edge_sizes``. Defaults to ``"short"``.
        interpolation_mode: Interpolation mode used for the resizing of the images.
            Defaults to ``"bilinear"``.

            .. note::
                For the resizing of guides ``"nearest"`` is used regardless of the
                ``interpolation_mode``.
        resize_targets: Targets for resizing of set images and guides during iteration.
    """

    def __init__(
        self,
        edge_sizes: Sequence[int],
        num_steps: Union[Sequence[int], int],
        edge: Union[Sequence[str], str] = "short",
        interpolation_mode: str = "bilinear",
        resize_targets: Collection[Union[Operator, MultiOperatorLoss]] = (),
    ):
        self._levels = self.build_levels(edge_sizes, num_steps, edge)
        self.interpolation_mode = interpolation_mode
        self._resize_targets = set(resize_targets)

    @staticmethod
    def build_levels(
        edge_sizes: Sequence[int],
        num_steps: Union[Sequence[int], int],
        edge: Union[Sequence[str], str],
    ) -> Tuple[PyramidLevel, ...]:
        num_levels = len(edge_sizes)
        if isinstance(num_steps, int):
            num_steps = [num_steps] * num_levels
        if isinstance(edge, str):
            edge = [edge] * num_levels

        return tuple(
            PyramidLevel(edge_size, num_steps_, edge_)
            for edge_size, num_steps_, edge_ in zip_equal(edge_sizes, num_steps, edge)
        )

    # TODO: can this be removed?
    def add_resize_target(self, op: Operator) -> None:
        self._resize_targets.add(op)

    def __len__(self) -> int:
        return len(self._levels)

    def __getitem__(self, idx: int) -> PyramidLevel:
        return self._levels[idx]

    def __iter__(self) -> Iterator[PyramidLevel]:
        image_storage = ImageStorage(self._resize_ops())
        for level in self._levels:
            try:
                self._resize(level)
                yield level
            finally:
                image_storage.restore()

    def _resize(self, level: PyramidLevel) -> None:
        for op in self._resize_ops():
            if isinstance(op, ComparisonOperator):
                if op.has_target_guide:
                    resized_guide = level.resize_guide(op.target_guide)
                    op.set_target_guide(resized_guide, recalc_repr=False)

                if op.has_target_image:
                    resized_image = level.resize_image(
                        op.target_image, interpolation_mode=self.interpolation_mode
                    )
                    op.set_target_image(resized_image)

            if op.has_input_guide:
                resized_guide = level.resize_guide(op.input_guide)
                op.set_input_guide(resized_guide)

    def _resize_ops(self) -> Set[Operator]:
        resize_ops = set()
        for target in self._resize_targets:
            if isinstance(target, Operator):
                resize_ops.add(target)

            for op in target.operators(recurse=True):
                if not isinstance(op, OperatorContainer):
                    resize_ops.add(op)
        return resize_ops

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        return dct

    def _named_children(self) -> Iterator[Tuple[str, Any]]:
        yield from super()._named_children()
        for idx, level in enumerate(self._levels):
            yield str(idx), level


class OctaveImagePyramid(ImagePyramid):
    r"""Image pyramid that comprises levels spaced by a factor of two.

    Args:
        max_edge_size: Maximum edge size.
        num_steps: Number of steps for each level.

            .. note::
                If ``num_steps`` is specified as sequence of ``int``s, you should also
                specify ``num_levels`` to match the lengths
        num_levels: Optional number of levels. If ``None``, the number is determined by
            the number of steps of factor two between ``max_edge_size`` and
            ``min_edge_size``.
        min_edge_size: Minimum edge size for the automatic calculation of
            ``num_levels``.
        image_pyramid_kwargs: Additional options. See
            :class:`~pystiche.pyramid.ImagePyramid` for details.
    """

    def __init__(
        self,
        max_edge_size: int,
        num_steps: Union[int, Sequence[int]],
        num_levels: Optional[int] = None,
        min_edge_size: int = 64,
        **image_pyramid_kwargs: Any,
    ) -> None:
        if num_levels is None:
            num_levels = int(np.floor(np.log2(max_edge_size / min_edge_size))) + 1

        edge_sizes = [
            round(max_edge_size / (2.0 ** ((num_levels - 1) - level)))
            for level in range(num_levels)
        ]
        super().__init__(edge_sizes, num_steps, **image_pyramid_kwargs)
