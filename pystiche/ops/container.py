from typing import Union, Sequence, Callable
from collections import OrderedDict
import torch
from pystiche.misc import build_obj_str
from pystiche.enc import Encoder, MultiLayerEncoder
from .op import Operator, EncodingOperator, ComparisonOperator
from .guidance import Guidance, ComparisonGuidance

__all__ = ["ContainerOperator", "MultiLayerEncodingOperator", "MultiRegionOperator"]


class ContainerOperator(Operator):
    def __init__(self, *args: Operator, score_weight=1e0) -> None:
        super().__init__(score_weight=score_weight)
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def process_input_image(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.score_weight * sum([op(input_image) for op in self.children()])

    def __getitem__(self, item: Union[str, int]):
        if isinstance(item, str):
            return self._get_children_by_name(item)
        elif isinstance(item, int):
            return self._get_children_by_index(item)
        else:
            raise TypeError

    def _get_children_by_name(self, name: str) -> Operator:
        children = dict(self.named_children())
        return children[name]

    def _get_children_by_index(self, idx: int) -> Operator:
        return tuple(self.children())[idx]


class MultiLayerEncodingOperator(ContainerOperator):
    def __init__(
        self,
        get_encoding_op: Callable[[Encoder, float], EncodingOperator],
        multi_layer_encoder: MultiLayerEncoder,
        layers: Sequence[str],
        layer_weights: Union[str, Sequence[float]] = "mean",
        score_weight: float = 1.0,
    ):
        layer_weights = self._parse_layer_weights(layer_weights, len(layers))
        ops = OrderedDict(
            [
                (layer, get_encoding_op(multi_layer_encoder[layer], layer_weight))
                for layer, layer_weight in zip(layers, layer_weights)
            ]
        )

        super().__init__(ops, score_weight=score_weight)

    @staticmethod
    def _parse_layer_weights(layer_weights, num_layers):
        if isinstance(layer_weights, str):
            if layer_weights == "mean":
                return [1.0 / num_layers] * num_layers
            elif layer_weights == "sum":
                return [1.0] * num_layers

            raise ValueError
        else:
            if len(layer_weights) == num_layers:
                return layer_weights

            raise ValueError

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


class MultiRegionOperator(ContainerOperator):
    def __init__(self, regions, get_op, score_weight=1e0):
        # FIXME: make a layer weights equivalent
        ops = OrderedDict([(region, get_op()) for region in regions])
        super().__init__(ops, score_weight=score_weight)

    def set_target_guide(self, region, guide):
        self[region].set_target_guide(guide)

    def set_target_image(self, region, image):
        self[region].set_target_image(image)

    def set_input_guide(self, region, guide):
        self[region].set_input_guide(guide)
