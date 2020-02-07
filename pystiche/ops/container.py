from typing import Union, Sequence, Dict, Callable
from collections import OrderedDict
import torch
import pystiche
from pystiche.enc import Encoder, MultiLayerEncoder
from .op import Operator, EncodingOperator, ComparisonOperator
from .guidance import Guidance, ComparisonGuidance

__all__ = ["Container", "MultiLayerEncodingOperator", "MultiRegionOperator"]


class Container(Operator):
    def __init__(self, named_ops: Dict[str, Operator], score_weight=1e0):
        super().__init__(score_weight=score_weight)
        for name, op in named_ops.items():
            self.add_module(name, op)

    def process_input_image(self, input_image: torch.Tensor) -> pystiche.LossDict:
        return pystiche.LossDict(
            [
                # (name, op(input_image) * self.score_weight)
                (name, op(input_image))
                for name, op in self.named_children()
            ]
        )

    def __getitem__(self, name):
        return self._modules[name]


class SameOperatorContainer(Container):
    def __init__(
        self,
        names: Sequence[str],
        get_op: Callable[[str, float], Operator],
        op_weights: Union[str, Sequence[float]] = "sum",
        score_weight=1e0,
    ) -> None:
        op_weights = self._parse_op_weights(op_weights, len(names))
        named_ops = OrderedDict(
            [(name, get_op(name, weight)) for name, weight in zip(names, op_weights)]
        )
        super().__init__(named_ops, score_weight=score_weight)

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
        layers: Sequence[str],
        get_encoding_op: Callable[[Encoder, float], EncodingOperator],
        multi_layer_encoder: MultiLayerEncoder,
        layer_weights: Union[str, Sequence[float]] = "mean",
        score_weight: float = 1e0,
    ):
        def get_op(layer, layer_weight):
            encoder = multi_layer_encoder[layer]
            return get_encoding_op(encoder, layer_weight)

        super().__init__(
            layers, get_op, op_weights=layer_weights, score_weight=score_weight
        )

    def set_target_guide(self, guide: torch.Tensor):
        for op in self.children():
            if isinstance(op, ComparisonGuidance):
                op.set_target_guide(guide)

    def set_target_image(self, image: torch.Tensor):
        for op in self.children():
            if isinstance(op, ComparisonOperator):
                op.set_target_image(image)

    def set_input_guide(self, guide: torch.Tensor):
        for op in self.children():
            if isinstance(op, Guidance):
                op.set_input_guide(guide)

    def __str__(self) -> str:
        def build_encoder_str():
            multi_layer_encoder = next(self.children()).encoder.multi_layer_encoder
            name = multi_layer_encoder.__class__.__name__
            properties = multi_layer_encoder.properties()
            named_children = ()
            return self._build_str(
                name=name, properties=properties, named_children=named_children
            )

        def build_op_str(op):
            properties = op.properties()
            del properties["encoder"]
            return op._build_str(properties=properties, named_children=())

        properties = OrderedDict()
        properties["encoder"] = build_encoder_str()
        properties.update(self.properties())

        named_children = [
            (name, build_op_str(op)) for name, op in self.named_children()
        ]

        return self._build_str(properties=properties, named_children=named_children)


class MultiRegionOperator(SameOperatorContainer):
    def __init__(
        self,
        regions: Sequence[str],
        get_op: Callable[[Encoder, float], Operator],
        region_weights: Union[str, Sequence[float]] = "sum",
        score_weight: float = 1e0,
    ):
        super().__init__(
            regions, get_op, op_weights=region_weights, score_weight=score_weight
        )

    def set_target_guide(self, region, guide):
        self[region].set_target_guide(guide)

    def set_target_image(self, region, image):
        self[region].set_target_image(image)

    def set_input_guide(self, region, guide):
        self[region].set_input_guide(guide)
