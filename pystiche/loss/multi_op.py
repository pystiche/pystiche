from typing import Iterator, Sequence, Tuple, Union

import torch

import pystiche
from pystiche.enc import MultiLayerEncoder, SingleLayerEncoder
from pystiche.misc import warn_deprecation
from pystiche.ops import EncodingOperator, Operator

__all__ = ["MultiOperatorLoss"]


class MultiOperatorLoss(pystiche.Module):
    def __init__(
        self, *named_ops: Sequence[Tuple[str, Operator]], trim: bool = True
    ) -> None:
        info = (
            "Please construct a MultiOperatorLoss with a sequence of named operators."
        )
        if len(named_ops) == 1:
            if isinstance(named_ops[0], dict):
                named_children = tuple(named_ops[0].items())
                warn_deprecation(
                    "Passing named_ops as dictionary", "0.4.0", info=info,
                )
            else:
                named_children = named_ops[0]
            indexed_children = None
        else:
            warn_deprecation(
                "Passing a variable number of unnamed operators via *args",
                "0.4.0",
                info=info,
            )
            named_children = None
            indexed_children = named_ops

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
            encoder.empty_storage()

        return loss

    def __getitem__(self, item: Union[str, int]):
        warn_deprecation(
            "Dynamic access to the modules via bracket indexing",
            "0.4.0",
            info="If you need dynamic access to the operators, use getattr() instead.",
        )
        if isinstance(item, str):
            return self._modules[item]
        elif isinstance(item, int):
            return self[self._get_child_name_by_idx(item)]
        else:
            raise TypeError

    def __delitem__(self, item: Union[str, int]):
        warn_deprecation("Deleting modules via bracket indexing", "0.4.0")
        if isinstance(item, str):
            del self._modules[item]
        elif isinstance(item, int):
            del self[self._get_child_name_by_idx(item)]
        else:
            raise TypeError

    def _get_child_name_by_idx(self, idx: int) -> str:
        return tuple(self._modules.keys())[idx]
