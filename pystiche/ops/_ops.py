from abc import abstractmethod
from typing import Any, Union, Optional, Tuple, Dict, Callable, Iterator, Sequence, List
from collections import OrderedDict
import torch

# import pystiche
from pystiche.typing import Numeric
from pystiche.misc import format_dict, to_engstr
from pystiche.enc import Encoder
from torch import nn

__all__ = [
    "Operator",
    "RegularizationOperator",
    "EncodingRegularizationOperator",
    "ComparisonOperator",
    "EncodingComparisonOperator",
    "MultiLayerEncodingOperator",
    "MultiLayerEncodingRegularizationOperator",
    "MultiLayerEncodingComparisonOperator",
]


class TensorStorage(nn.Module):
    def __init__(self, **attrs):
        super().__init__()
        for name, attr in attrs.items():
            if isinstance(attr, torch.Tensor):
                self.register_buffer(name, attr)
            else:
                setattr(self, name, attr)

    def forward(self):
        msg = (
            f"{self.__class__.__name__} objects are only used "
            "for storage and cannot be called."
        )
        raise RuntimeError(msg)


class Operator(nn.Module):
    def __init__(self, score_weight: float = 1.0) -> None:
        super().__init__()
        self.score_weight = score_weight

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.process_input_image(input_image) * self.score_weight

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class RegularizationOperator(Operator):
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        input_repr = self.input_image_to_repr(image)
        return self.calculate_score(input_repr)

    @abstractmethod
    def input_image_to_repr(
        self, image: torch.Tensor
    ) -> Union[torch.Tensor, TensorStorage]:
        pass

    @abstractmethod
    def calculate_score(
        self, input_repr: Union[torch.Tensor, TensorStorage]
    ) -> torch.Tensor:
        pass


class EncodingRegularizationOperator(RegularizationOperator):
    def __init__(self, encoder: Encoder, layer: str, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight)
        self.encoder = encoder
        self.layer = layer

    def image_to_enc(self, image):
        return self.encoder(image, layers=(self.layer,))[0]

    def input_image_to_repr(
        self, image: torch.Tensor
    ) -> Union[torch.Tensor, TensorStorage]:
        return self.image_to_enc(image)

    @abstractmethod
    def input_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Union[torch.Tensor, TensorStorage]:
        pass

    @abstractmethod
    def calculate_score(
        self, input_repr: Union[torch.Tensor, TensorStorage]
    ) -> torch.Tensor:
        pass


class ComparisonOperator(Operator):
    # def __init__(self, score_weight: float = 1.0):
    #     super().__init__(score_weight=score_weight)
    #     self.target_image = None
    #     self.target_repr = None
    #     self.ctx = None

    def set_target_image(self, image: torch.Tensor):
        with torch.no_grad():
            repr, ctx = self.target_image_to_repr(image)
        # self.target_image = image.detach()
        # self.target_repr = repr.detach()
        # self.ctx = ctx.detach() if ctx is not None else ctx
        self.register_buffer("target_image", image)
        self.register_buffer("target_repr", repr)
        self.register_buffer("ctx", ctx)

    @property
    def has_target_image(self) -> bool:
        return self.target_image is not None

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.has_target_image:
            # TODO: message
            raise RuntimeError

        target_repr, ctx = self.target_repr, self.ctx
        input_repr = self.input_image_to_repr(image, ctx)
        return self.calculate_score(input_repr, target_repr, ctx)

    @abstractmethod
    def input_image_to_repr(
        self, image: torch.Tensor, ctx: Optional[Union[torch.Tensor, TensorStorage]]
    ) -> Union[torch.Tensor, TensorStorage]:
        pass

    @abstractmethod
    def target_image_to_repr(
        self, image: torch.Tensor
    ) -> Tuple[
        Union[torch.Tensor, TensorStorage], Optional[Union[torch.Tensor, TensorStorage]]
    ]:
        pass

    @abstractmethod
    def calculate_score(
        self,
        input_repr: Union[torch.Tensor, TensorStorage],
        target_repr: Union[torch.Tensor, TensorStorage],
        ctx: Optional[Union[torch.Tensor, TensorStorage]],
    ) -> torch.Tensor:
        pass


class EncodingComparisonOperator(ComparisonOperator):
    def __init__(self, encoder: Encoder, layer: str, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight)
        self.encoder = encoder
        self.layer = layer

    def image_to_enc(self, image):
        return self.encoder(image, layers=(self.layer,))[0]

    def input_image_to_repr(
        self, image: torch.Tensor, ctx: Optional[Union[torch.Tensor, TensorStorage]]
    ) -> Union[torch.Tensor, TensorStorage]:
        enc = self.image_to_enc(image)
        return self.input_enc_to_repr(enc, ctx)

    def target_image_to_repr(
        self, image: torch.Tensor
    ) -> Tuple[
        Union[torch.Tensor, TensorStorage], Optional[Union[torch.Tensor, TensorStorage]]
    ]:
        enc = self.image_to_enc(image)
        return self.target_enc_to_repr(enc)

    @abstractmethod
    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[Union[torch.Tensor, TensorStorage]]
    ) -> Union[torch.Tensor, TensorStorage]:
        pass

    @abstractmethod
    def target_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Tuple[
        Union[torch.Tensor, TensorStorage], Optional[Union[torch.Tensor, TensorStorage]]
    ]:
        pass

    @abstractmethod
    def calculate_score(
        self,
        input_repr: Union[torch.Tensor, TensorStorage],
        target_repr: Union[torch.Tensor, TensorStorage],
        ctx: Optional[Union[torch.Tensor, TensorStorage]],
    ) -> torch.Tensor:
        pass


# move to container
class MultiLayerEncodingOperator(Operator):
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
        super().__init__(score_weight=score_weight)
        layer_weights = self._verify_layer_weights(layer_weights, layers)

        for idx, (layer, layer_weight) in enumerate(zip(layers, layer_weights)):
            op = get_encoding_op(layer, layer_weight)
            self.add_module(str(idx), op)

    @property
    def layers(self) -> List[str]:
        return [op.layer for op in self.encoding_operators()]

    @abstractmethod
    def encoding_operators(
            self
    ) -> Iterator[Union[EncodingRegularizationOperator, EncodingComparisonOperator]]:
        pass

    def process_input_image(self, image: torch.Tensor):
        # a = []
        # for op in self.encoding_operators():
        #     loss = op(image)
        #     a.append(loss)
        # return sum(a)
        return sum([op(image) for op in self.encoding_operators()])

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




class MultiLayerEncodingRegularizationOperator(MultiLayerEncodingOperator):
    # FIXME: override the type annotations for __init__()

    def encoding_operators(self):
        for module in self.children():
            if isinstance(module, EncodingRegularizationOperator):
                yield module


class MultiLayerEncodingComparisonOperator(MultiLayerEncodingOperator):
    # FIXME: override the type annotations for __init__() and operators()

    def encoding_operators(self):
        for module in self.children():
            if isinstance(module, EncodingComparisonOperator):
                yield module

    def set_target_image(self, image: torch.Tensor):
        for op in self.encoding_operators():
            op.set_target_image(image)


# class Operator(TensorStorage):
#     def __init__(self, name: str, score_weight: Numeric = 1.0):
#         super().__init__()
#         self.name = name
#         self.score_weight = score_weight
#
#     def __call__(self, input_image: torch.Tensor) -> torch.Tensor:
#         return self._process_input_image(input_image) * self.score_weight
#
#     def extra_str(self) -> str:
#         dct = self._descriptions()
#         dct.update(self.extra_descriptions())
#         return format_dict(dct, sep=" : ")
#
#     def _descriptions(self) -> Dict[str, Any]:
#         dct = OrderedDict()
#         dct["Name"] = self.name
#         dct["Score weight"] = to_engstr(self.score_weight)
#         return dct
#
#     def extra_descriptions(self) -> Dict[str, Any]:
#         return OrderedDict()
#
#     @abstractmethod
#     def _process_input_image(self, image: torch.Tensor) -> torch.Tensor:
#         pass
#
#
# class RegularizationOperator(Operator):
#     def _process_input_image(self, image: torch.Tensor) -> torch.Tensor:
#         input_repr = self._input_image_to_repr(image)
#         return self._calculate_score(input_repr)
#
#     @abstractmethod
#     def _input_image_to_repr(
#         self, image: torch.Tensor
#     ) -> Union[torch.Tensor, TensorStorage]:
#         pass
#
#     @abstractmethod
#     def _calculate_score(
#         self, input_repr: Union[torch.Tensor, TensorStorage]
#     ) -> torch.Tensor:
#         pass
#
#
# class EncodingRegularizationOperator(RegularizationOperator):
#     def __init__(
#         self, name: str, encoder: Encoder, layer: str, score_weight: float = 1.0
#     ):
#         super().__init__(name, score_weight)
#         self.encoder = encoder
#         self.layer = layer
#
#     def _descriptions(self) -> Dict[str, Any]:
#         dct = OrderedDict()
#         dct["Name"] = self.name
#         # FIXME
#         # dct["Encoder"] = to_engstr(self.score_weight)
#         dct["Layer"] = self.layer
#         dct["Score weight"] = to_engstr(self.score_weight)
#         return dct
#
#     def _image_to_enc(self, image):
#         return self.encoder(image, (self.layer,))[0]
#
#     def _input_image_to_repr(
#         self, image: torch.Tensor
#     ) -> Union[torch.Tensor, TensorStorage]:
#         return self._image_to_enc(image)
#
#     @abstractmethod
#     def _input_enc_to_repr(
#         self, enc: torch.Tensor
#     ) -> Union[torch.Tensor, TensorStorage]:
#         pass
#
#     @abstractmethod
#     def _calculate_score(
#         self, input_repr: Union[torch.Tensor, TensorStorage]
#     ) -> torch.Tensor:
#         pass
#
#
#
#
# class ComparisonOperator(Operator):
#     def __init__(self, name: str, score_weight: float = 1.0):
#         super().__init__(name, score_weight)
#         self.target_image = None
#         self._target_repr = None
#         self._ctx = None
#
#     def set_target_image(self, image: torch.Tensor):
#         with torch.no_grad():
#             repr, ctx = self._target_image_to_repr(image)
#         self.target_image = image.detach()
#         self._target_repr = repr.detach()
#         if ctx is not None:
#             ctx = ctx.detach()
#         self._ctx = ctx
#
#     @property
#     def has_target_image(self) -> bool:
#         return self.target_image is not None
#
#     def _process_input_image(self, image: torch.Tensor) -> torch.Tensor:
#         if not self.has_target_image:
#             # TODO: message
#             raise RuntimeError
#
#         target_repr, ctx = self._target_repr, self._ctx
#         input_repr = self._input_image_to_repr(image, ctx)
#         return self._calculate_score(input_repr, target_repr, ctx)
#
#     @abstractmethod
#     def _input_image_to_repr(
#         self,
#         image: torch.Tensor,
#         ctx: Optional[Union[torch.Tensor, TensorStorage]],
#     ) -> Union[torch.Tensor, TensorStorage]:
#         pass
#
#     @abstractmethod
#     def _target_image_to_repr(
#         self, image: torch.Tensor
#     ) -> Tuple[
#         Union[torch.Tensor, TensorStorage],
#         Optional[Union[torch.Tensor, TensorStorage]],
#     ]:
#         pass
#
#     @abstractmethod
#     def _calculate_score(
#         self,
#         input_repr: Union[torch.Tensor, TensorStorage],
#         target_repr: Union[torch.Tensor, TensorStorage],
#         ctx: Optional[Union[torch.Tensor, TensorStorage]],
#     ) -> torch.Tensor:
#         pass
#
#
# class EncodingComparisonOperator(ComparisonOperator):
#     def __init__(
#         self, name: str, encoder: Encoder, layer: str, score_weight: float = 1.0
#     ):
#         super().__init__(name, score_weight)
#         self.encoder = encoder
#         self.layer = layer
#
#     def _descriptions(self) -> Dict[str, Any]:
#         dct = OrderedDict()
#         dct["Name"] = self.name
#         # FIXME
#         # dct["Encoder"] = to_engstr(self.score_weight)
#         dct["Layer"] = self.layer
#         dct["Score weight"] = to_engstr(self.score_weight)
#         return dct
#
#     def _image_to_enc(self, image):
#         return self.encoder(image, (self.layer,))[0]
#
#     def _input_image_to_repr(
#         self,
#         image: torch.Tensor,
#         ctx: Optional[Union[torch.Tensor, TensorStorage]],
#     ) -> Union[torch.Tensor, TensorStorage]:
#         enc = self._image_to_enc(image)
#         return self._input_enc_to_repr(enc, ctx)
#
#     def _target_image_to_repr(
#         self, image: torch.Tensor
#     ) -> Tuple[
#         Union[torch.Tensor, TensorStorage],
#         Optional[Union[torch.Tensor, TensorStorage]],
#     ]:
#         enc = self._image_to_enc(image)
#         return self._target_enc_to_repr(enc)
#
#     @abstractmethod
#     def _input_enc_to_repr(
#         self,
#         enc: torch.Tensor,
#         ctx: Optional[Union[torch.Tensor, TensorStorage]],
#     ) -> Union[torch.Tensor, TensorStorage]:
#         pass
#
#     @abstractmethod
#     def _target_enc_to_repr(
#         self, enc: torch.Tensor
#     ) -> Tuple[
#         Union[torch.Tensor, TensorStorage],
#         Optional[Union[torch.Tensor, TensorStorage]],
#     ]:
#         pass
#
#     @abstractmethod
#     def _calculate_score(
#         self,
#         input_repr: Union[torch.Tensor, TensorStorage],
#         target_repr: Union[torch.Tensor, TensorStorage],
#         ctx: Optional[Union[torch.Tensor, TensorStorage]],
#     ) -> torch.Tensor:
#         pass
