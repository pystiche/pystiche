from collections import OrderedDict
from typing import Callable, Dict, Sequence, Tuple, Union, cast

import torch
from torch import nn

import pystiche
from pystiche.enc import MultiLayerEncoder, SingleLayerEncoder
from pystiche.misc import verify_str_arg

from .op import ComparisonOperator, EncodingOperator, Operator

__all__ = [
    "OperatorContainer",
    "SameOperatorContainer",
    "MultiLayerEncodingOperator",
    "MultiRegionOperator",
]


class OperatorContainer(Operator):
    r"""Generic container for :class:`~pystiche.ops.Operator` s. If called with an image
    passes it to all immediate operators and returns a :class:`pystiche.LossDict`
    scaled with ``score_weight``.

    Args:
        named_ops: Named immediate operators that will be called if
            :class:`OperatorContainer` is called.
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def __init__(
        self, named_ops: Sequence[Tuple[str, Operator]], score_weight: float = 1e0,
    ) -> None:
        super().__init__(score_weight=score_weight)
        self.add_named_modules(named_ops)

    def process_input_image(self, input_image: torch.Tensor) -> pystiche.LossDict:
        return pystiche.LossDict(
            [(name, op(input_image)) for name, op in self.named_children()]
        )

    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        r"""Invoke :meth:`~pystiche.ops.ComparisonOperator.set_target_guide` on all
        immediate :class:`~pystiche.ops.ComparisonOperator` children.

        Args:
            guide: Input guide of shape :math:`1 \times 1 \times H \times W`.
            recalc_repr: If ``True``, recalculates :meth:`.target_enc_to_repr`.
                Defaults to ``True``.
        """
        for op in self.operators():
            if isinstance(op, ComparisonOperator):
                op.set_target_guide(guide, recalc_repr=recalc_repr)

    def set_target_image(self, image: torch.Tensor) -> None:
        r"""Invoke :meth:`~pystiche.ops.ComparisonOperator.set_target_image` on all
        immediate :class:`~pystiche.ops.ComparisonOperator` children.

        Args:
            image: Target image of shape :math:`B \times C \times H \times W`.
        """
        for op in self.operators():
            if isinstance(op, ComparisonOperator):
                op.set_target_image(image)

    def set_input_guide(self, guide: torch.Tensor) -> None:
        r"""Invoke :meth:`~pystiche.ops.Operator.set_input_guide` on all immediate
        children.

        Args:
            guide: Input guide of shape :math:`1 \times 1 \times H \times W`.
        """
        for op in self.operators():
            op.set_input_guide(guide)

    _modules: Dict[str, nn.Module]


class SameOperatorContainer(OperatorContainer):
    def __init__(
        self,
        names: Sequence[str],
        get_op: Callable[[str, float], Operator],
        op_weights: Union[str, Sequence[float]] = "sum",
        score_weight: float = 1e0,
    ) -> None:
        op_weights = self._parse_op_weights(op_weights, len(names))
        named_ops = [
            (name, get_op(name, weight)) for name, weight in zip(names, op_weights)
        ]

        super().__init__(named_ops, score_weight=score_weight)

    @staticmethod
    def _parse_op_weights(
        op_weights: Union[str, Sequence[float]], num_ops: int
    ) -> Sequence[float]:
        if isinstance(op_weights, str):
            verify_str_arg(op_weights, "op_weights", ("sum", "mean"))
            if op_weights == "sum":
                return [1.0] * num_ops
            else:  # op_weights == "mean":
                return [1.0 / num_ops] * num_ops
        else:
            if len(op_weights) == num_ops:
                return op_weights

            msg = (
                f"The length of the operator weights and the number of operators do "
                f"not match: {len(op_weights)} != {num_ops}"
            )
            raise ValueError(msg)


class MultiLayerEncodingOperator(SameOperatorContainer):
    r"""Convenience container for multiple :class:`~pystiche.ops.EncodingOperator` s
    operating on different ``layers`` of the same ``multi_layer_encoder``.

    Args:
        multi_layer_encoder: Multi-layer encoder.
        layers: Layers of the ``multi_layer_encoder`` that the children operators
            operate on.
        get_encoding_op: Callable that returns a children operator given a
            :class:`pystiche.enc.SingleLayerEncoder` extracted from the
            ``multi_layer_encoder`` and its corresponding layer weight.
        layer_weights: Weights of the children operators passed to ``get_encoding_op``.
            If ``"sum"``, each layer weight is set to ``1.0``. If ``"mean"``, each
            layer weight is set to ``1.0 / len(layers)``. If sequence of ``float``s its
            length has to match ``layers``. Defaults to ``"mean"``.
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def __init__(
        self,
        multi_layer_encoder: MultiLayerEncoder,
        layers: Sequence[str],
        get_encoding_op: Callable[[SingleLayerEncoder, float], EncodingOperator],
        layer_weights: Union[str, Sequence[float]] = "mean",
        score_weight: float = 1e0,
    ):
        def get_op(layer: str, layer_weight: float) -> EncodingOperator:
            encoder = multi_layer_encoder.extract_encoder(layer)
            return get_encoding_op(encoder, layer_weight)

        super().__init__(
            layers, get_op, op_weights=layer_weights, score_weight=score_weight,
        )

    def __repr__(self) -> str:
        def build_encoder_repr() -> str:
            encoding_op = cast(EncodingOperator, next(self.operators()))
            multi_layer_encoder = cast(
                MultiLayerEncoder, encoding_op.encoder.multi_layer_encoder
            )
            name = multi_layer_encoder.__class__.__name__
            properties = multi_layer_encoder.properties()
            named_children = ()
            return self._build_repr(
                name=name, properties=properties, named_children=named_children
            )

        def build_op_repr(op: Operator) -> str:
            properties = op.properties()
            del properties["encoder"]
            return op._build_repr(properties=properties, named_children=())

        properties = OrderedDict()
        properties["encoder"] = build_encoder_repr()
        properties.update(self.properties())

        named_children = [
            (name, build_op_repr(op)) for name, op in self.named_operators()
        ]

        return self._build_repr(properties=properties, named_children=named_children)


class MultiRegionOperator(SameOperatorContainer):
    r"""Convenience container for multiple :class:`~pystiche.ops.Operator` s
    operating in different ``regions``.

    Args:
        regions: Regions.
        get_op: Callable that returns a children operator given a region its
            corresponding region weight.
        region_weights: Weights of the children operators passed to ``get_op``. If
            ``"sum"``, each region weight is set to ``1.0``. If ``"mean"``, each region
            weight is set to ``1.0 / len(layers)``. If sequence of ``float``s its
            length has to match ``regions``. Defaults to ``"mean"``.
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def __init__(
        self,
        regions: Sequence[str],
        get_op: Callable[[str, float], Operator],
        region_weights: Union[str, Sequence[float]] = "sum",
        score_weight: float = 1e0,
    ):
        super().__init__(
            regions, get_op, op_weights=region_weights, score_weight=score_weight,
        )

    def set_regional_target_guide(self, region: str, guide: torch.Tensor) -> None:
        r"""Invokes :meth:`~pystiche.ops.Comparison.set_target_guide` on the operator
        of the given ``region``.

        Args:
            region: Region.
            guide: Input guide of shape :math:`1 \times 1 \times H \times W`.
        """
        getattr(self, region).set_target_guide(guide)

    def set_regional_target_image(self, region: str, image: torch.Tensor) -> None:
        r"""Invokes :meth:`~pystiche.ops.Comparison.set_target_image` on the operator
        of the given ``region``.

        Args:
            region: Region.
            image: Input guide of shape :math:`B \times C \times H \times W`.
        """
        getattr(self, region).set_target_image(image)

    def set_regional_input_guide(self, region: str, guide: torch.Tensor) -> None:
        r"""Invokes :meth:`~pystiche.ops.Comparison.set_input_guide` on the operator
        of the given ``region``.

        Args:
            region: Region.
            guide: Input guide of shape :math:`1 \times 1 \times H \times W`.
        """
        getattr(self, region).set_input_guide(guide)
