from collections import OrderedDict
import torch
import pystiche
from pystiche.loss.multi_op_encoder import MultiOperatorEncoder
from pystiche.ops import Operator
from .loss_dict import LossDict

__all__ = ["MultiOperatorLoss"]


class MultiOperatorLoss(pystiche.object):
    def __init__(self, *ops: Operator, trim: bool = True) -> None:
        super().__init__()
        self._ops = pystiche.tuple(OrderedDict.fromkeys(ops))
        self._multi_op_encoders = self._collect_multi_op_encoders()

        if trim:
            for encoder in self._multi_op_encoders:
                encoder.trim()

    def _collect_multi_op_encoders(self):
        # FIXME: automatically detect same encoder on multiple ops and create the
        # FIXME: MultiOperatorEncoder

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

    def extra_str(self):
        return "\n".join([str(op) for op in self._ops])
