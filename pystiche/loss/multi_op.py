from typing import Iterator, Sequence, Tuple

import torch

import pystiche
from pystiche.enc import MultiLayerEncoder, SingleLayerEncoder
from pystiche.ops import EncodingOperator, Operator

__all__ = ["MultiOperatorLoss"]


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
        self, named_ops: Sequence[Tuple[str, Operator]], trim: bool = True
    ) -> None:
        super().__init__(named_children=named_ops,)

        self._multi_layer_encoders = self._collect_multi_layer_encoders()

        if trim:
            for encoder in self._multi_layer_encoders:
                encoder.trim()

    def named_operators(self, recurse: bool = False) -> Iterator[Tuple[str, Operator]]:
        iterator = self.named_modules() if recurse else self.named_children()
        for name, child in iterator:
            if isinstance(child, Operator):
                yield name, child

    def operators(self, recurse: bool = False) -> Iterator[Operator]:
        for _, op in self.named_operators(recurse=recurse):
            yield op

    def _collect_multi_layer_encoders(self) -> Tuple[MultiLayerEncoder, ...]:
        def encoding_ops() -> Iterator[Operator]:
            for op in self.operators(recurse=True):
                if isinstance(op, EncodingOperator):
                    yield op

        multi_layer_encoders = set()
        for op in encoding_ops():
            if isinstance(op.encoder, SingleLayerEncoder):
                multi_layer_encoders.add(op.encoder.multi_layer_encoder)

        return tuple(multi_layer_encoders)

    def forward(self, input_image: torch.Tensor) -> pystiche.LossDict:
        for encoder in self._multi_layer_encoders:
            encoder.encode(input_image)

        loss = pystiche.LossDict(
            [(name, op(input_image)) for name, op in self.named_children()]
        )

        for encoder in self._multi_layer_encoders:
            encoder.empty_storage()

        return loss
