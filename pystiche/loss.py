from typing import Union, Tuple, Dict, Iterator
from collections import OrderedDict
import torch
import pystiche
from pystiche.enc import SingleLayerEncoder, MultiLayerEncoder
from pystiche.ops import Operator, EncodingOperator

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

        self._multi_layer_encoders = self._collect_multi_layer_encoders()

        if trim:
            for encoder in self._multi_layer_encoders:
                encoder.trim()

    def _collect_multi_layer_encoders(self) -> Tuple[MultiLayerEncoder, ...]:
        def encoding_ops() -> Iterator[EncodingOperator]:
            for module in self.modules():
                if isinstance(module, EncodingOperator):
                    yield module

        multi_layer_encoders = set()
        for op in encoding_ops():
            encoder = op.encoder
            if isinstance(encoder, SingleLayerEncoder):
                multi_layer_encoders.add(encoder.multi_layer_encoder)

        return tuple(multi_layer_encoders)

    def forward(self, input_image: torch.Tensor) -> pystiche.LossDict:
        for encoder in self._multi_layer_encoders:
            encoder.encode(input_image)

        loss = pystiche.LossDict(
            [(name, op(input_image)) for name, op in self.named_children()]
        )

        for encoder in self._multi_layer_encoders:
            encoder.clear_cache()

        return loss

    def __getitem__(self, item: Union[str, int]):
        if isinstance(item, str):
            return self._modules[item]
        elif isinstance(item, int):
            return self[self._get_child_name_by_idx(item)]
        else:
            raise TypeError

    def __delitem__(self, item: Union[str, int]):
        if isinstance(item, str):
            del self._modules[item]
        elif isinstance(item, int):
            del self[self._get_child_name_by_idx(item)]
        else:
            raise TypeError

    def _get_child_name_by_idx(self, idx: int) -> str:
        return tuple(self._modules.keys())[idx]
