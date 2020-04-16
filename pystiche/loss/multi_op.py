import warnings
from typing import Iterator, Sequence, Tuple, Union

import torch

import pystiche
from pystiche.enc import MultiLayerEncoder, SingleLayerEncoder
from pystiche.misc import build_deprecation_message
from pystiche.ops import Operator
from pystiche.ops.meta import Latent

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
                msg = build_deprecation_message(
                    "Passing named_ops as dictionary", "0.4.0", info=info,
                )
                warnings.warn(msg)
            else:
                named_children = named_ops[0]
            indexed_children = None
        else:
            msg = build_deprecation_message(
                "Passing a variable number of unnamed operators via *args",
                "0.4.0",
                info=info,
            )
            warnings.warn(msg)
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
        def latent_ops() -> Iterator[Operator]:
            for op in self.modules():
                if isinstance(op, Operator) and isinstance(op.domain, Latent):
                    yield op

        multi_layer_encoders = set()
        for op in latent_ops():
            try:
                encoder = op.encoder
                if isinstance(encoder, SingleLayerEncoder):
                    multi_layer_encoders.add(encoder.multi_layer_encoder)
            except AttributeError:
                pass

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
        msg = build_deprecation_message(
            "Dynamic access to the modules via bracket indexing",
            "0.4.0",
            info="If you need dynamic access to the operators, use getattr() instead.",
        )
        warnings.warn(msg)
        if isinstance(item, str):
            return self._modules[item]
        elif isinstance(item, int):
            return self[self._get_child_name_by_idx(item)]
        else:
            raise TypeError

    def __delitem__(self, item: Union[str, int]):
        msg = build_deprecation_message(
            "Deleting modules via bracket indexing", "0.4.0"
        )
        warnings.warn(msg)
        if isinstance(item, str):
            del self._modules[item]
        elif isinstance(item, int):
            del self[self._get_child_name_by_idx(item)]
        else:
            raise TypeError

    def _get_child_name_by_idx(self, idx: int) -> str:
        return tuple(self._modules.keys())[idx]
