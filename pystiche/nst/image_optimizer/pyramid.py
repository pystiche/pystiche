from typing import Union, Optional, Sequence, Dict, Callable
import numpy as np
import torch
import pystiche
from pystiche.misc import zip_equal, verify_str_arg
from pystiche.image import extract_image_size, extract_aspect_ratio
from pystiche.image.transforms import FixedAspectRatioResize, GrayscaleToBinary
from ..operators import Operator, Comparison, Guidance, ComparisonGuidance
from .image_optimizer import ImageOptimizer

__all__ = ["PyramidLevel", "ImageOptimizerPyramid", "ImageOptimizerOctavePyramid"]


class PyramidLevel(pystiche.object):
    def __init__(
        self,
        num: int,
        edge_size: int,
        num_steps: int,
        edge: str,
        binarizer: Optional[Callable] = None,
        interpolation_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.num: int = num
        self.edge_size = edge_size
        self.num_steps: int = num_steps
        self.edge = verify_str_arg(edge, "edge", ("short", "long"))

        if binarizer is None:
            binarizer = GrayscaleToBinary()
        self.binarizer = binarizer

        self.interpolation_mode = interpolation_mode

    def resize(
        self,
        image: torch.Tensor,
        aspect_ratio: Optional[float] = None,
        binarize: bool = False,
    ):
        transform = FixedAspectRatioResize(
            self.edge_size,
            self.edge,
            aspect_ratio=aspect_ratio,
            interpolation_mode=self.interpolation_mode,
        )
        image = transform(image)
        if binarize:
            image = self.binarizer(image)
        return image

    def extra_str(self) -> str:
        extras = [
            "num={num}",
            "edge_size={edge_size}",
            "num_steps={num_steps}",
            "edge={edge}",
        ]
        if self.interpolation_mode != "bilinear":
            extras.append("interpolation_mode={interpolation_mode}")
        return ", ".join(extras).format(size=self.transform.size, **self.__dict__)


class ImageOptimizerPyramid(pystiche.object):
    InitialState = pystiche.namedtuple(
        "init_state", ("target_image", "input_guide", "target_guide")
    )

    def __init__(self, image_optimizer: ImageOptimizer):
        super().__init__()
        self.image_optimizer: ImageOptimizer = image_optimizer
        self._levels = None

    def build_levels(
        self,
        level_edge_sizes: Sequence[int],
        level_steps: Union[Sequence[int], int],
        edges: Union[Sequence[str], str] = "short",
        **kwargs,
    ):
        num_levels = len(level_edge_sizes)
        if isinstance(level_steps, int):
            level_steps = [level_steps] * num_levels
        if isinstance(edges, str):
            edges = [edges] * num_levels

        levels = [
            PyramidLevel(num, *level_args, **kwargs)
            for num, level_args in enumerate(
                zip_equal(level_edge_sizes, level_steps, edges)
            )
        ]
        self._levels = pystiche.tuple(levels)

    @property
    def has_levels(self) -> bool:
        return self._levels is not None

    def _assert_has_levels(self):
        if not self.has_levels:
            msg = "You need to call build_levels() before starting the optimization"
            raise RuntimeError(msg)

    def max_resize(self, image, **kwargs):
        self._assert_has_levels()
        return self._levels[-1].resize(image, **kwargs)

    def __call__(self, input_image: torch.Tensor, quiet: bool = False, **kwargs):
        self._assert_has_levels()
        init_states = self._extract_operator_initial_states()
        output_images = self._iterate(input_image, init_states, quiet, **kwargs)
        self._reset_operators(init_states)
        return pystiche.tuple(output_images).detach()

    def _extract_operator_initial_states(self) -> Dict[Operator, InitialState]:
        operators = tuple(self.image_optimizer.operators())
        init_states = []
        for operator in operators:
            has_input_guide = (
                isinstance(operator, Guidance) and operator.has_input_guide
            )
            input_guide = operator.input_guide if has_input_guide else None

            has_target_guide = (
                isinstance(operator, ComparisonGuidance) and operator.has_target_guide
            )
            target_guide = operator.target_guide if has_target_guide else None

            has_target_image = (
                isinstance(operator, Comparison) and operator.has_target_image
            )
            target_image = operator.target_image if has_target_image else None

            init_states.append(
                self.InitialState(target_image, input_guide, target_guide)
            )
        return dict(zip(operators, init_states))

    def _reset_operators(self, init_states: Dict[Operator, InitialState]):
        for operator, init_state in init_states.items():
            if isinstance(operator, Guidance):
                operator.set_input_guide(init_state.input_guide)

            if isinstance(operator, ComparisonGuidance):
                operator.set_target_guide(init_state.target_guide)

            if isinstance(operator, Comparison):
                operator.set_target(init_state.target_image)

    def _iterate(
        self,
        input_image: torch.Tensor,
        init_states: InitialState,
        quiet: bool,
        **kwargs,
    ):
        aspect_ratio = extract_aspect_ratio(input_image)
        output_images = [input_image]
        for level in self._levels:
            input_image = level.resize(output_images[-1], aspect_ratio=aspect_ratio)
            self._resize_operator_images(level, init_states)

            if not quiet:
                self._print_header(level.num, input_image)

            output_image = self.image_optimizer(
                input_image, level.num_steps, quiet=quiet, **kwargs
            )
            output_images.append(output_image)

        return pystiche.tuple(output_images[1:])

    def _resize_operator_images(
        self, level: PyramidLevel, init_states: Dict[Operator, InitialState]
    ):
        for operator, init_state in init_states.items():
            if isinstance(operator, Guidance):
                if init_state.input_guide is None:
                    continue
                guide = level.resize(init_state.input_guide, binarize=True)
                operator.set_input_guide(guide)

            if isinstance(operator, ComparisonGuidance):
                if init_state.target_guide is None:
                    continue
                guide = level.resize(init_state.target_guide, binarize=True)
                operator.set_target_guide(guide)

            if isinstance(operator, Comparison):
                if init_state.target_image is None:
                    continue
                image = level.resize(init_state.target_image)
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
        max_edge_size: int,
        level_steps: Union[Sequence[int], int],
        num_levels: Optional[int] = None,
        min_edge_size: int = 64,
        edges: Union[Sequence[str], str] = "short",
        **kwargs,
    ):
        if num_levels is None:
            num_levels = int(np.floor(np.log2(max_edge_size / min_edge_size))) + 1

        level_edge_sizes = [
            round(max_edge_size / (2.0 ** ((num_levels - 1) - level)))
            for level in range(num_levels)
        ]
        super().build_levels(level_edge_sizes, level_steps, edges=edges)
