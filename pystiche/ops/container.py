from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

import pystiche
from pystiche.enc import Encoder, MultiLayerEncoder

from . import meta
from .op import EncodingOperator, Operator

__all__ = [
    "OperatorContainer",
    "SameOperatorContainer",
    "MultiLayerEncodingOperator",
    "MultiRegionOperator",
]


class OperatorContainer(Operator):
    def __init__(
        self,
        named_ops: Sequence[Tuple[str, Operator]],
        cls: Optional[Union[meta.OperatorCls, str]] = None,
        domain: Optional[Union[meta.OperatorDomain, str]] = None,
        score_weight=1e0,
    ):
        super().__init__(cls=cls, domain=domain, score_weight=score_weight)
        self.add_named_modules(named_ops)

    def process_input_image(self, input_image: torch.Tensor) -> pystiche.LossDict:
        return pystiche.LossDict(
            [(name, op(input_image)) for name, op in self.named_children()]
        )

    # def set_target_guide(self, guide: torch.Tensor):
    #     for op in self.children():
    #         if isinstance(op, ComparisonGuidance):
    #             op.set_target_guide(guide)

    def set_target_image(self, image: torch.Tensor, recurse: bool = True):
        for op in self.operators(recurse=recurse):
            if op is self:
                continue

            if isinstance(op.cls, meta.Comparison):
                try:
                    op.set_target_image(image)
                except AttributeError:
                    pass

    # def set_input_guide(self, guide: torch.Tensor):
    #     for op in self.children():
    #         if isinstance(op, Guidance):
    #             op.set_input_guide(guide)

    # TODO: can this be removed?
    def __getitem__(self, name):
        return self._modules[name]


class SameOperatorContainer(OperatorContainer):
    def __init__(
        self,
        names: Sequence[str],
        get_op: Callable[[str, float], Operator],
        op_weights: Union[str, Sequence[float]] = "sum",
        cls: Optional[Union[meta.OperatorCls, str]] = None,
        domain: Optional[Union[meta.OperatorDomain, str]] = None,
        score_weight=1e0,
    ) -> None:
        op_weights = self._parse_op_weights(op_weights, len(names))
        named_ops = [
            (name, get_op(name, weight)) for name, weight in zip(names, op_weights)
        ]

        super().__init__(named_ops, cls=cls, domain=domain, score_weight=score_weight)

    @staticmethod
    def _parse_op_weights(op_weights, num_ops):
        if isinstance(op_weights, str):
            if op_weights == "sum":
                return [1.0] * num_ops
            elif op_weights == "mean":
                return [1.0 / num_ops] * num_ops

            raise ValueError
        else:
            if len(op_weights) == num_ops:
                return op_weights

            raise ValueError


class MultiLayerEncodingOperator(SameOperatorContainer):
    def __init__(
        self,
        multi_layer_encoder: MultiLayerEncoder,
        layers: Sequence[str],
        get_encoding_op: Callable[[Encoder, float], EncodingOperator],
        layer_weights: Union[str, Sequence[float]] = "mean",
        cls: Optional[Union[meta.OperatorCls, str]] = None,
        domain: Optional[Union[meta.OperatorDomain, str]] = meta.Encoding(),
        score_weight: float = 1e0,
    ):
        def get_op(layer, layer_weight):
            encoder = multi_layer_encoder.extract_single_layer_encoder(layer)
            return get_encoding_op(encoder, layer_weight)

        super().__init__(
            layers,
            get_op,
            op_weights=layer_weights,
            cls=cls,
            domain=domain,
            score_weight=score_weight,
        )

    def __repr__(self) -> str:
        def build_encoder_repr():
            multi_layer_encoder = next(self.children()).encoder.multi_layer_encoder
            name = multi_layer_encoder.__class__.__name__
            properties = multi_layer_encoder.properties()
            named_children = ()
            return self._build_repr(
                name=name, properties=properties, named_children=named_children
            )

        def build_op_repr(op):
            properties = op.properties()
            del properties["encoder"]
            return op._build_repr(properties=properties, named_children=())

        properties = OrderedDict()
        properties["encoder"] = build_encoder_repr()
        properties.update(self.properties())

        named_children = [
            (name, build_op_repr(op)) for name, op in self.named_children()
        ]

        return self._build_repr(properties=properties, named_children=named_children)


class MultiRegionOperator(SameOperatorContainer):
    def __init__(
        self,
        regions: Sequence[str],
        get_op: Callable[[str, float], Operator],
        region_weights: Union[str, Sequence[float]] = "sum",
        cls: Optional[Union[meta.OperatorCls, str]] = None,
        domain: Optional[Union[meta.OperatorDomain, str]] = None,
        score_weight: float = 1e0,
    ):
        super().__init__(
            regions,
            get_op,
            op_weights=region_weights,
            cls=cls,
            domain=domain,
            score_weight=score_weight,
        )

    # def set_target_guide(self, region, guide):
    #     self[region].set_target_guide(guide)

    def set_regional_target_image(self, region: str, image: torch.Tensor) -> None:
        getattr(self, region).set_target_image(image)
        self[region].set_target_image(image)

    # def set_input_guide(self, region, guide):
    #     self[region].set_input_guide(guide)
