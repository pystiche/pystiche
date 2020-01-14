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
from pystiche.misc import subclass_iterator
from pystiche.image.transforms import (
    TorchPreprocessing,
    TorchPostprocessing,
    CaffePreprocessing,
    CaffePostprocessing,
)
from pystiche.enc import Encoder
from pystiche.nst.encoder import MultiOperatorEncoder
from pystiche.nst.operators import (
    Operator,
    ComparisonOperator,
    EncodingOperator,
    PixelOperator,
)
from .loss_dict import LossDict

__all__ = [
    "MultiOperatorLoss",
    # "PreprocessingImageOptimizer",
    # "TorchPreprocessingImageOptimizer",
    # "CaffePreprocessingImageOptimizer",
]


class MultiOperatorLoss(pystiche.object):
    def __init__(self, *ops: Operator, trim: bool = True) -> None:
        super().__init__()
        self._ops = pystiche.tuple(OrderedDict.fromkeys(ops))
        self._multi_op_encoders = self._collect_multi_op_encoders()

        if trim:
            for encoder in self._multi_op_encoders:
                encoder.trim()

    def _collect_multi_op_encoders(self):
        # FIXME: rename
        def blub():
            for op in self._ops:
                try:
                    multi_op_encoder = op.encoder
                except AttributeError:
                    continue
                if not isinstance(multi_op_encoder, MultiOperatorEncoder):
                    continue

                yield op, multi_op_encoder

        multi_op_encoders = set()
        for op, multi_op_encoder in blub():
            multi_op_encoders.add(multi_op_encoder)

        for encoder in multi_op_encoders:
            encoder.reset_layers()

        for op, multi_op_encoder in blub():
            multi_op_encoder.register_layers(op.layers)

        return pystiche.tuple(multi_op_encoders)

    def __call__(self, input_image: torch.Tensor) -> LossDict:

        for encoder in self._multi_op_encoders:
            encoder.encode(input_image)

        loss = LossDict([(op.name, op(input_image)) for op in self._ops])

        for encoder in self._multi_op_encoders:
            encoder.clear_storage()

        return loss

    # def _iterate(
    #     self,
    #     input_image: torch.Tensor,
    #     num_steps: int,
    # ) -> torch.Tensor:
    #     optimizer = self.optimizer_getter(input_image.requires_grad_(True))
    #     for step in range(1, num_steps + 1):
    #         self._optimize(input_image, optimizer)
    #         self._diagnose(input_image)
    #     return input_image
    #
    # def _optimize(self, input_image: torch.Tensor, optimizer: Optimizer):
    #     optimizer.step(lambda: self._closure(input_image, optimizer))
    #
    # def _closure(self, input_image: torch.Tensor, optimizer: Optimizer) -> torch.Tensor:
    #     optimizer.zero_grad()
    #     for encoder in self.multi_operator_encoders():
    #         encoder.encode(input_image)
    #
    #     loss = sum(
    #         [
    #             operator(input_image)
    #             for operator in self.operators()
    #         ]
    #     )
    #     loss.backward()
    #     return loss
    #
    # def operators(self) -> Iterator[Operator]:
    #     # FIXME: is this needed?
    #     for operator in self._operators:
    #         yield operator
    #
    # def encoders(
    #     self, *args: Any, **kwargs: Any
    # ) -> Iterator[Union[Encoder, MultiOperatorEncoder]]:
    #     return subclass_iterator(self._encoders, *args, **kwargs)
    #
    # def multi_operator_encoders(self) -> Iterator[MultiOperatorEncoder]:
    #     return self.encoders(MultiOperatorEncoder)

    def extra_str(self):
        return "\n".join([str(op) for op in self._ops])


# class PreprocessingImageOptimizer(ImageOptimizer):
#     def __init__(
#         self, *args: Any, multi_encoder_warning: bool = True, **kwargs: Any
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         if multi_encoder_warning and len(self._encoders) > 1:
#             msg = (
#                 "Multiple encoders detected. Are you sure that you want to use "
#                 "multiple encoders with the same preprocessing? To suppress this "
#                 "warning, set multi_encoder_warning=False."
#             )
#             warnings.warn(msg, RuntimeWarning)
#
#     @abstractmethod
#     def preprocess(self, image: torch.Tensor) -> torch.Tensor:
#         pass
#
#     @abstractmethod
#     def postprocess(self, image: torch.Tensor) -> torch.Tensor:
#         pass
#
#     def __call__(
#         self, input_image: torch.Tensor, *args: Any, **kwargs: Any
#     ) -> torch.Tensor:
#         input_image = self.preprocess(input_image)
#         operators = tuple(self.operators(EncodingOperator, ComparisonOperator))
#         target_images = []
#         for operator in operators:
#             target_image = operator.target_image
#             target_images.append(target_image)
#             operator.set_target(self.preprocess(target_image))
#
#         output_image = super().__call__(input_image, *args, **kwargs)
#
#         for operator, target_image in zip(operators, target_images):
#             operator.target_image = target_image
#         output_image = self.postprocess(output_image)
#
#         return output_image
#
#     def _closure(self, input_image: torch.Tensor, optimizer: Optimizer) -> torch.Tensor:
#         optimizer.zero_grad()
#         for encoder in self.multi_operator_encoders():
#             encoder.encode(input_image)
#
#         operators = set(self.operators())
#         loss = torch.tensor(0.0, **pystiche.tensor_meta(input_image))
#
#         pixel_operators = set(
#             [operator for operator in operators if isinstance(operator, PixelOperator)]
#         )
#         if pixel_operators:
#             input_image_postprocessed = self.postprocess(input_image)
#             for operator in pixel_operators:
#                 loss += operator(input_image_postprocessed)
#
#         encoding_operators = operators - pixel_operators
#         for operator in encoding_operators:
#             loss += operator(input_image)
#
#         loss.backward()
#         return loss
#
#
# class TorchPreprocessingImageOptimizer(PreprocessingImageOptimizer):
#     def __init__(self, *args: Any, **kwargs: Any) -> None:
#         super().__init__(*args, **kwargs)
#         self.preprocessing_transform = TorchPreprocessing()
#         self.postprocessing_transform = TorchPostprocessing()
#
#     def preprocess(self, image: torch.Tensor) -> torch.Tensor:
#         return self.preprocessing_transform(image)
#
#     def postprocess(self, image: torch.Tensor) -> torch.Tensor:
#         return self.postprocessing_transform(image)
#
#
# class CaffePreprocessingImageOptimizer(PreprocessingImageOptimizer):
#     def __init__(self, *args: Any, **kwargs: Any) -> None:
#         super().__init__(*args, **kwargs)
#         self.preprocessing_transform = CaffePreprocessing()
#         self.postprocessing_transform = CaffePostprocessing()
#
#     def preprocess(self, image: torch.Tensor) -> torch.Tensor:
#         return self.preprocessing_transform(image)
#
#     def postprocess(self, image: torch.Tensor) -> torch.Tensor:
#         return self.postprocessing_transform(image)
