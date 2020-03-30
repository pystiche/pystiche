from typing import Union, Tuple, Dict, Iterator
import torch
import pystiche
from pystiche.enc import SingleLayerEncoder, MultiLayerEncoder
from pystiche.ops import (
    Operator,
    EncodingOperator,
)
from pystiche.misc import warn_deprecation

__all__ = ["MultiOperatorLoss"]


class MultiOperatorLoss(pystiche.Module):
    def __init__(
        self, *args: Union[Dict[str, Operator], Operator], trim: bool = True
    ) -> None:
        if len(args) == 1 and isinstance(args[0], dict):
            named_children = args[0]
            indexed_children = None
        else:
            warn_deprecation(
                "variable number of input",
                "*args",
                "0.4",
                info="Please construct a MultiOperatorLoss with a dictionary of named operators.",
            )
            named_children = None
            indexed_children = args

        super().__init__(
            named_children=named_children, indexed_children=indexed_children
        )

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
        warn_deprecation(
            "method",
            "__getitem__",
            "0.4",
            info="If you need dynamic access to the operators, use getattr() instead.",
        )
        if isinstance(item, str):
            return self._modules[item]
        elif isinstance(item, int):
            return self[self._get_child_name_by_idx(item)]
        else:
            raise TypeError

    def __delitem__(self, item: Union[str, int]):
        warn_deprecation("method", "__delitem__", "0.4")
        if isinstance(item, str):
            del self._modules[item]
        elif isinstance(item, int):
            del self[self._get_child_name_by_idx(item)]
        else:
            raise TypeError

    def _get_child_name_by_idx(self, idx: int) -> str:
        return tuple(self._modules.keys())[idx]
