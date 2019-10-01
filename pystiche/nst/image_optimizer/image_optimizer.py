from abc import abstractmethod
from typing import Any, Optional, Union, Callable, Iterator, Iterable, Sequence
import warnings
from collections import OrderedDict
import itertools
from math import floor
import torch
from torch import optim
from torch.optim.optimizer import Optimizer
import pystiche
from pystiche.image.transforms import (
    TorchPreprocessing,
    TorchPostprocessing,
    CaffePreprocessing,
    CaffePostprocessing,
)
from pystiche.encoding import Encoder
from ..encoder import MultiOperatorEncoder
from ..operators import Operator, Comparison, Encoding, Pixel, Diagnosis

__all__ = [
    "ImageOptimizer",
    "PreprocessingImageOptimizer",
    "TorchPreprocessingImageOptimizer",
    "CaffePreprocessingImageOptimizer",
]


class ImageOptimizer(pystiche.object):
    def __init__(
        self, *operators: Operator, optimizer_getter: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self._operators = pystiche.tuple(OrderedDict.fromkeys(operators))

        encoders = [operator.encoder for operator in self.operators(Encoding)]
        self._encoders = pystiche.set(encoders)

        if optimizer_getter is None:

            def optimizer_getter(input_image):
                return optim.LBFGS([input_image], lr=1.0, max_iter=1)

        self.optimizer_getter = optimizer_getter

    def __call__(
        self,
        input_image: torch.Tensor,
        num_steps: int,
        trim: bool = True,
        quiet=False,
        print_steps: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        if print_steps is None:
            print_steps = max((round(num_steps / 1e2) * 10, 10))
        if isinstance(print_steps, int):
            print_steps = self._interval_to_steps(print_steps, num_steps)

        maxlen_name_str = max([operator.len_name_str for operator in self.operators()])
        for operator in self.operators():
            operator.len_name_str = maxlen_name_str

        for encoder in self.nst_encoders():
            encoder.reset_layers()

        for operator in self.operators(Encoding):
            encoder = operator.encoder
            if isinstance(encoder, MultiOperatorEncoder):
                encoder.register_layers(operator.layers)

        if trim:
            for encoder in self.nst_encoders():
                encoder.trim()

        output_image = self._iterate(input_image, num_steps, quiet, print_steps)

        for encoder in self.nst_encoders():
            encoder.clear_storage()

        return output_image.detach()

    @staticmethod
    def _interval_to_steps(interval: int, num_steps: int) -> Iterable[int]:
        num_intervals = int(floor(num_steps / interval))
        steps = set([interval * step for step in range(1, num_intervals + 1)])
        steps.add(num_steps)
        return steps

    def _iterate(
        self,
        input_image: torch.Tensor,
        num_steps: int,
        quiet: bool,
        print_steps: Iterable[int],
    ) -> torch.Tensor:
        optimizer = self.optimizer_getter(input_image.requires_grad_(True))
        for step in range(1, num_steps + 1):
            self._optimize(input_image, optimizer)
            self._diagnose(input_image)
            if not quiet and step in print_steps:
                self._print_scores(step)
        return input_image

    def _optimize(self, input_image: torch.Tensor, optimizer: Optimizer):
        optimizer.step(lambda: self._closure(input_image, optimizer))

    def _closure(self, input_image: torch.Tensor, optimizer: Optimizer) -> torch.Tensor:
        optimizer.zero_grad()
        for encoder in self.nst_encoders():
            encoder.encode(input_image)

        loss = sum(
            [
                operator(input_image)
                for operator in self.operators(Diagnosis, not_instance=True)
            ]
        )
        loss.backward()
        return loss

    def _diagnose(self, input_image: torch.Tensor):
        for operator in self.operators(Diagnosis):
            operator(input_image)

    def _print_scores(self, step: int):
        for operator in self.operators():
            operator.print_score(step)
        print("-" * operator.len_score_str)

    @staticmethod
    def _subclass_iterator(
        sequence: Sequence,
        *subclasses: Any,
        not_instance: bool = False,
        all_subclasses: bool = True
    ) -> Iterator[Any]:
        if not subclasses:
            return iter(sequence)

        mask = [
            [isinstance(obj, subclass) for subclass in subclasses] for obj in sequence
        ]
        reduce_fn = all if all_subclasses else any
        mask = map(lambda x: not_instance ^ reduce_fn(x), mask)
        return itertools.compress(sequence, mask)

    def operators(self, *args: Any, **kwargs: Any) -> Iterator[Operator]:
        return self._subclass_iterator(self._operators, *args, **kwargs)

    def encoders(
        self, *args: Any, **kwargs: Any
    ) -> Iterator[Union[Encoder, MultiOperatorEncoder]]:
        return self._subclass_iterator(self._encoders, *args, **kwargs)

    def nst_encoders(self) -> Iterator[MultiOperatorEncoder]:
        return self.encoders(MultiOperatorEncoder)

    def extra_str(self):
        return "\n".join([str(operator) for operator in self.operators()])


class PreprocessingImageOptimizer(ImageOptimizer):
    def __init__(
        self, *args: Any, multi_encoder_warning: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if multi_encoder_warning and len(self._encoders) > 1:
            msg = ("Multiple encoders detected. Are you sure that you want to use "
                   "multiple encoders with the same preprocessing? To suppress this "
                   "warning, set multi_encoder_warning=False.")
            warnings.warn(msg, RuntimeWarning)

    @abstractmethod
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def postprocess(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(
        self, input_image: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        input_image = self.preprocess(input_image)
        operators = tuple(self.operators(Encoding, Comparison))
        target_images = []
        for operator in operators:
            target_image = operator.target_image
            target_images.append(target_image)
            operator.set_target(self.preprocess(target_image))

        output_image = super().__call__(input_image, *args, **kwargs)

        for operator, target_image in zip(operators, target_images):
            operator.target_image = target_image
        output_image = self.postprocess(output_image)

        return output_image

    def _closure(self, input_image: torch.Tensor, optimizer: Optimizer) -> torch.Tensor:
        optimizer.zero_grad()
        for encoder in self.nst_encoders():
            encoder.encode(input_image)

        operators = set(self.operators(Diagnosis, not_instance=True))
        loss = torch.tensor(0.0, **pystiche.tensor_meta(input_image))

        pixel_operators = set(
            [operator for operator in operators if isinstance(operator, Pixel)]
        )
        if pixel_operators:
            input_image_postprocessed = self.postprocess(input_image)
            for operator in pixel_operators:
                loss += operator(input_image_postprocessed)

        encoding_operators = operators - pixel_operators
        for operator in encoding_operators:
            loss += operator(input_image)

        loss.backward()
        return loss

    def _diagnose(self, input_image: torch.Tensor):
        operators = set(self.operators(Diagnosis))
        if not operators:
            return

        pixel_operators = set(
            [operator for operator in operators if isinstance(operator, Pixel)]
        )
        if pixel_operators:
            input_image_postprocessed = self.postprocess(input_image)
            for operator in pixel_operators:
                operator(input_image_postprocessed)

        encoding_operators = operators - pixel_operators
        for operator in encoding_operators:
            operator(input_image)


class TorchPreprocessingImageOptimizer(PreprocessingImageOptimizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.preprocessing_transform = TorchPreprocessing()
        self.postprocessing_transform = TorchPostprocessing()

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return self.preprocessing_transform(image)

    def postprocess(self, image: torch.Tensor) -> torch.Tensor:
        return self.postprocessing_transform(image)


class CaffePreprocessingImageOptimizer(PreprocessingImageOptimizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.preprocessing_transform = CaffePreprocessing()
        self.postprocessing_transform = CaffePostprocessing()

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return self.preprocessing_transform(image)

    def postprocess(self, image: torch.Tensor) -> torch.Tensor:
        return self.postprocessing_transform(image)
