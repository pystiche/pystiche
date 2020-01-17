from typing import Union, Sequence, Callable
from collections import OrderedDict
import torch
from .op import Operator, EncodingRegularizationOperator, EncodingComparisonOperator


__all__ = [
    "CompundOperator",
    "MultiLayerEncodingOperator",
    "MultiLayerEncodingComparisonOperator",
]


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
        get_encoding_op: Callable[
            [str, float],
            Union[EncodingRegularizationOperator, EncodingComparisonOperator],
        ],
        layers: Sequence[str],
        layer_weights: Union[str, Sequence[float]] = "mean",
        score_weight: float = 1.0,
    ):
        ops = self._create_ops(get_encoding_op, layers, layer_weights)
        super().__init__(*ops, score_weight=score_weight)

    def _create_ops(self, get_encoding_op, layers, layer_weights):
        layer_weights = self._verify_layer_weights(layer_weights, layers)
        return [
            get_encoding_op(layer, layer_weight)
            for layer, layer_weight in zip(layers, layer_weights)
        ]

    @staticmethod
    def _verify_layer_weights(layer_weights, layers):
        num_layers = len(layers)
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


class MultiLayerEncodingComparisonOperator(MultiLayerEncodingOperator):
    # FIXME: override the type annotations for __init__()

    def set_target_image(self, image: torch.Tensor):
        for op in self.children():
            if isinstance(op, EncodingComparisonOperator):
                op.set_target_image(image)
