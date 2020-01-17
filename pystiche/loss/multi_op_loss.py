from collections import OrderedDict
import torch
import pystiche
from pystiche.loss.multi_op_encoder import MultiOperatorEncoder
from pystiche.ops import Operator
from .loss_dict import LossDict

from torch import nn
__all__ = ["MultiOperatorLoss"]

from ..ops import EncodingRegularizationOperator, EncodingComparisonOperator


class MultiOperatorLoss(nn.Module):
    def __init__(self, *ops: Operator, trim: bool = True) -> None:
        super().__init__()
        for idx, op in enumerate(ops):
            self.add_module(str(idx), op)

        self._encoders = self._collect_encoders()

        if trim:
            for encoder in self.encoders():
                encoder.trim()

    def _collect_encoders(self):
        encoders = set()
        for op in self.operators():
            try:
                encoder = op.encoder
            except AttributeError:
                continue

            encoder.register_layer(op.layer)
            encoders.add(encoder)

        return tuple(encoders)

    def forward(self, input_image: torch.Tensor) -> LossDict:
        for encoder in self.encoders():
            encoder.encode(input_image)

        # a = []
        # for name, op in self.named_operators():
        #     loss = op(input_image)
        #     a.append((name, loss))
        #
        # return LossDict(a)

        return LossDict([(name, op(input_image)) for name, op in self.named_operators()])


    def named_operators(self):
        for name, module in self.named_modules():
            if isinstance(module, Operator):
                yield name, module

    def operators(self):
        for name, op in self.named_operators():
            yield op

    def encoders(self):
        for encoder in self._encoders:
            yield encoder
