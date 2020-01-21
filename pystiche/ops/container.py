from typing import Union, Sequence, Callable
from collections import OrderedDict
import torch
from pystiche.enc import Encoder, MultiLayerEncoder
from .op import Operator, EncodingOperator, ComparisonOperator
from .guidance import Guidance, ComparisonGuidance


__all__ = ["CompundOperator", "MultiLayerEncodingOperator"]


class CompundOperator(Operator):
    def __init__(self, *args: Operator, score_weight=1.0) -> None:
        super().__init__(score_weight=score_weight)
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def process_input_image(self, input_image: torch.Tensor) -> torch.Tensor:
        return sum([op(input_image) for op in self.children()])


class MultiLayerEncodingOperator(CompundOperator):
    def __init__(
        self,
        get_encoding_op: Callable[[Encoder, float], EncodingOperator],
        multi_layer_encoder: MultiLayerEncoder,
        layers: Sequence[str],
        layer_weights: Union[str, Sequence[float]] = "mean",
        score_weight: float = 1.0,
    ):
        layer_weights = self._parse_layer_weights(layer_weights, len(layers))
        ops = [
            get_encoding_op(multi_layer_encoder[layer], layer_weight)
            for layer, layer_weight in zip(layers, layer_weights)
        ]
        super().__init__(*ops, score_weight=score_weight)

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

    def set_target_guide(self, image: torch.Tensor):
        for op in self.children():
            if isinstance(op, ComparisonGuidance):
                op.set_target_guide(image)

    def set_target_image(self, image: torch.Tensor):
        for op in self.children():
            if isinstance(op, ComparisonOperator):
                op.set_target_image(image)

    def set_input_guide(self, image: torch.Tensor):
        for op in self.children():
            if isinstance(op, Guidance):
                op.set_input_guide(image)
