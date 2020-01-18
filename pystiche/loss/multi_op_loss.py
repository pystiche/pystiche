from typing import Union, Tuple, Dict
from collections import OrderedDict
import torch
import pystiche
from pystiche.enc import MultiLayerEncoder
from pystiche.ops import Operator
from .loss_dict import LossDict


__all__ = ["MultiOperatorLoss"]


class MultiOperatorLoss(pystiche.Module):
    def __init__(
        self, *args: Union[Dict[str, Operator], Operator], trim: bool = True
    ) -> None:
        super().__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

        self._encoders = self._collect_encoders()

        if trim:
            for encoder in self._encoders:
                encoder.trim()

    def _collect_encoders(self) -> Tuple[MultiLayerEncoder, ...]:
        encoders = set()
        for op in self.modules():
            # FIXME: replace by isinstance(op, EncodingOp)
            # FIXME: afterwards: maybe remove method?
            try:
                encoder = op.encoder
            except AttributeError:
                continue

            encoder.register_layer(op.layer)
            encoders.add(encoder)

        return tuple(encoders)

    def forward(self, input_image: torch.Tensor) -> LossDict:
        for encoder in self._encoders:
            encoder.encode(input_image)

        loss = LossDict([(name, op(input_image)) for name, op in self.named_children()])

        for encoder in self._encoders:
            encoder.clear_storage()

        return loss
