from types import TracebackType
from typing import Iterator, Sequence, Tuple, Type

import torch
from torch import nn

import pystiche
from pystiche import enc, ops
from pystiche.misc import suppress_warnings

__all__ = ["MLEHandler", "MultiOperatorLoss"]


class MLEHandler(pystiche.ComplexObject):
    def __init__(self, criterion: nn.Module) -> None:
        self.multi_layer_encoders = {
            encoding_op.encoder.multi_layer_encoder
            for encoding_op in criterion.modules()
            if isinstance(encoding_op, ops.EncodingOperator)
            and isinstance(encoding_op.encoder, enc.SingleLayerEncoder)
        }

    def encode(self, input_image: torch.Tensor) -> None:
        for encoder in self.multi_layer_encoders:
            encoder.encode(input_image)

    def empty_storage(self) -> None:
        for mle in self.multi_layer_encoders:
            mle.empty_storage()

    def trim(self) -> None:
        for mle in self.multi_layer_encoders:
            mle.trim()

    def __call__(self, input_image: torch.Tensor) -> "MLEHandler":
        with suppress_warnings(FutureWarning):
            for mle in self.multi_layer_encoders:
                mle.encode(input_image)
        return self

    def __enter__(self) -> None:
        pass

    def __exit__(
        self, exc_type: Type[Exception], exc_val: Exception, exc_tb: TracebackType
    ) -> None:
        for encoder in self.multi_layer_encoders:
            encoder.empty_storage()

    def _named_children(self) -> Iterator[Tuple[str, enc.MultiLayerEncoder]]:
        return ((str(idx), mle) for idx, mle in enumerate(self.multi_layer_encoders))


class MultiOperatorLoss(pystiche.Module):
    r"""Generic loss for multiple :class:`~pystiche.ops.Operator` s. If called with an
    image it is passed to all immediate children operators and the
    results are returned as a :class:`pystiche.LossDict`. For that each
    :class:`pystiche.enc.MultiLayerEncoder` is only hit once, even if it is associated
    with multiple of the called operators.

    Args:
        named_ops: Named children operators.
        trim: If ``True``, all :class:`~pystiche.enc.MultiLayerEncoder` s associated
            with ``content_loss``, ``style_loss``, or ``regularization`` will be
            :meth:`~pystiche.enc.MultiLayerEncoder.trim` med. Defaults to ``True``.
    """

    def __init__(
        self, named_ops: Sequence[Tuple[str, ops.Operator]], trim: bool = True
    ) -> None:
        super().__init__(named_children=named_ops)
        self._mle_handler = MLEHandler(self)

        if trim:
            self._mle_handler.trim()

    def named_operators(
        self, recurse: bool = False
    ) -> Iterator[Tuple[str, ops.Operator]]:
        iterator = self.named_modules() if recurse else self.named_children()
        for name, child in iterator:
            if isinstance(child, ops.Operator):
                yield name, child

    def operators(self, recurse: bool = False) -> Iterator[ops.Operator]:
        for _, op in self.named_operators(recurse=recurse):
            yield op

    def forward(self, input_image: torch.Tensor) -> pystiche.LossDict:
        with self._mle_handler(input_image):
            return pystiche.LossDict(
                [(name, op(input_image)) for name, op in self.named_children()]
            )
