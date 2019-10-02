import warnings
from typing import Union, Optional, Sequence, Tuple, Dict, Callable
import numpy as np
import torch
import pystiche
from pystiche.misc import zip_equal
from pystiche.image import (
    is_image_size,
    is_edge_size,
    calculate_aspect_ratio,
    image_to_edge_size,
    extract_image_size,
)
from pystiche.image.transforms import (
    Transform,
    ResizeTransform,
    Resize,
    FixedAspectRatioResize,
    GrayscaleToBinary,
)
from ..operators import Operator, Comparison, Guidance, ComparisonGuidance
from .image_optimizer import ImageOptimizer

__all__ = ["PyramidLevel", "ImageOptimizerPyramid", "ImageOptimizerOctavePyramid"]


class PyramidLevel(pystiche.object):
    def __init__(
        self,
        num: int,
        num_steps: int,
        transform: Callable,
        guide_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.num: int = num
        self.num_steps: int = num_steps

        if isinstance(transform, ResizeTransform) and not transform.has_fixed_size:
            msg = (
                "The usage of a resize transformation that calculates the image size "
                "at runtime is not recommended. If you experience size-mismatch "
                "errors, consider using resize transformations with a fixed size."
            )
            warnings.warn(msg, RuntimeWarning)
        self.transform: Callable = transform

        if guide_transform is None and isinstance(transform, Transform):
            guide_transform = transform + GrayscaleToBinary()
        self.guide_transform: Callable = guide_transform

    def extra_str(self) -> str:
        extra = "num={num}", "num_steps={num_steps}", "size={size}"
        return ", ".join(extra).format(size=self.transform.size, **self.__dict__)


class ImageOptimizerPyramid(pystiche.object):
    InitialState = pystiche.namedtuple(
        "init_state", ("target_image", "input_guide", "target_guide")
    )

    def __init__(self, image_optimizer: ImageOptimizer):
        super().__init__()
        self.image_optimizer: ImageOptimizer = image_optimizer
        self._levels = None

    def build_levels(
        self, level_image_sizes, level_steps: Union[Sequence[int], int], **kwargs
    ):
        if isinstance(level_steps, int):
            level_steps = tuple([level_steps] * len(level_image_sizes))

        level_transforms = [
            Resize(level_image_size)
            if is_image_size(level_image_size)
            else FixedAspectRatioResize(level_image_size, **kwargs)
            for level_image_size in level_image_sizes
        ]

        levels = [
            PyramidLevel(num, num_steps, transform)
            for num, (num_steps, transform) in enumerate(
                zip_equal(level_steps, level_transforms)
            )
        ]
        self._levels = pystiche.tuple(levels)

    @property
    def has_levels(self) -> bool:
        return self._levels is not None

    def assert_has_levels(self):
        if not self.has_levels:
            # TODO: add error message
            raise RuntimeError

    @property
    def max_level_transform(self) -> Callable:
        self.assert_has_levels()
        return self._levels[-1].transform

    @property
    def max_level_guide_transform(self) -> Callable:
        self.assert_has_levels()
        return self._levels[-1].guide_transform

    def __call__(self, input_image: torch.Tensor, quiet: bool = False, **kwargs):
        self.assert_has_levels()

        init_states = self._extract_comparison_initial_states()

        output_images = self._iterate(input_image, init_states, quiet, **kwargs)

        return pystiche.tuple(output_images).detach()

    def _extract_comparison_initial_states(self) -> Dict[Operator, InitialState]:
        operators = tuple(self.image_optimizer.operators())
        init_states = []
        for operator in operators:
            has_target_image = (
                isinstance(operator, Comparison) and operator.has_target_image
            )
            target_image = operator.target_image if has_target_image else None

            has_input_guide = (
                isinstance(operator, Guidance) and operator.has_input_guide
            )
            input_guide = operator.input_guide if has_input_guide else None

            has_target_guide = (
                isinstance(operator, ComparisonGuidance) and operator.has_target_guide
            )
            target_guide = operator.target_guide if has_target_guide else None

            init_states.append(
                self.InitialState(target_image, input_guide, target_guide)
            )
        return dict(zip(operators, init_states))

    def _iterate(
        self,
        input_image: torch.Tensor,
        init_states: InitialState,
        quiet: bool,
        **kwargs
    ):
        output_images = [input_image]
        for level in self._levels:
            input_image = level.transform(output_images[-1])
            self._transform_targets(level.transform, level.guide_transform, init_states)

            if not quiet:
                self._print_header(level.num, input_image)

            output_image = self.image_optimizer(
                input_image, level.num_steps, quiet=quiet, **kwargs
            )
            output_images.append(output_image)

        return pystiche.tuple(output_images[1:])

    def _transform_targets(
        self,
        transform: Callable,
        guide_transform: Callable,
        init_states: Dict[Operator, InitialState],
    ):
        for operator, init_state in init_states.items():
            if isinstance(operator, Guidance) and init_state.input_guide is not None:
                guide = guide_transform(init_state.input_guide)
                operator.set_input_guide(guide)

            if (
                isinstance(operator, ComparisonGuidance)
                and init_state.target_guide is not None
            ):
                guide = guide_transform(init_state.target_guide)
                operator.set_target_guide(guide)

            image = transform(init_state.target_image)
            operator.set_target(image)

    def _print_header(self, level: int, image: torch.Tensor):
        image_size = extract_image_size(image)
        line = " Pyramid level {0} ({2} x {1}) ".format(level, *reversed(image_size))
        sep_line = "=" * max((len(line), 39))
        print(sep_line)
        print(line)
        print(sep_line)


class ImageOptimizerOctavePyramid(ImageOptimizerPyramid):
    def build_levels(
        self,
        size: Union[Tuple[int, int], int],
        level_steps: Union[Sequence[int], int],
        num_levels: Optional[int] = None,
        min_edge_size: int = 64,
        edge: str = "short",
    ):
        edge_size, aspect_ratio = self._extract_image_params(size, edge)

        if num_levels is None:
            num_levels = int(np.floor(np.log2(edge_size / min_edge_size))) + 1

        level_image_sizes = [
            round(edge_size / (2.0 ** ((num_levels - 1) - level)))
            for level in range(num_levels)
        ]
        super().build_levels(
            level_image_sizes, level_steps, aspect_ratio=aspect_ratio, edge=edge
        )

    @staticmethod
    def _extract_image_params(size: Union[Tuple[int, int], int], edge: str):
        if is_image_size(size):
            return image_to_edge_size(size, edge), calculate_aspect_ratio(size)
        elif is_edge_size(size):
            return size, None
        else:
            # FIXME: error message
            raise ValueError
