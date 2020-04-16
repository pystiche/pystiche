import warnings
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch

import pystiche
from pystiche.enc import Encoder
from pystiche.misc import build_deprecation_message, is_almost, to_engstr

from . import meta

__all__ = [
    "Operator",
    "RegularizationOperator",
    "ComparisonOperator",
    "PixelOperator",
    "EncodingOperator",
    "PixelRegularizationOperator",
    "EncodingRegularizationOperator",
    "PixelComparisonOperator",
    "EncodingComparisonOperator",
]


class Operator(pystiche.Module):
    def __init__(
        self,
        cls: Optional[Union[meta.OperatorCls, str]] = None,
        domain: Optional[Union[meta.OperatorDomain, str]] = None,
        score_weight: float = 1e0,
    ) -> None:
        super().__init__()
        self._cls = meta.cls(cls)
        self._domain = meta.domain(domain)
        self.score_weight = score_weight

    @property
    def cls(self) -> meta.OperatorCls:
        return self._cls

    @property
    def domain(self) -> meta.OperatorDomain:
        return self._domain

    def forward(
        self, input_image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.LossDict]:
        return self.process_input_image(input_image) * self.score_weight

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def named_operators(
        self, recurse: bool = False
    ) -> Iterator[Tuple[str, "Operator"]]:
        if recurse:
            iterator = self.named_modules()
        else:
            iterator = self.named_children()
        for name, child in iterator:
            if isinstance(child, Operator):
                yield name, child

    # FIXME: recurse might be wrong phrase here
    def operators(self, recurse: bool = False) -> Iterator["Operator"]:
        for _, op in self.named_operators(recurse=recurse):
            yield op

    def _properties(self) -> Dict[str, Any]:
        dct = OrderedDict()
        if not is_almost(self.score_weight, 1e0):
            dct["score_weight"] = to_engstr(self.score_weight)
        return dct


class RegularizationOperator(Operator):
    def __init__(self, score_weight: float = 1e0):
        msg = build_deprecation_message(
            "The ABC RegularizationOperator",
            "0.4.0",
            info=(
                "The same behavior can be achieved by passing "
                "cls=pystiche.ops.meta.Regularization() to Operator."
            ),
        )
        warnings.warn(msg)
        super().__init__(cls=meta.Regularization(), score_weight=score_weight)

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class ComparisonOperator(Operator):
    def __init__(self, score_weight: float = 1e0):
        msg = build_deprecation_message(
            "The ABC ComparisonOperator",
            "0.4.0",
            info=(
                "The same behavior can be achieved by passing "
                "cls=pystiche.ops.meta.Comparison() to Operator."
            ),
        )
        warnings.warn(msg)
        super().__init__(cls=meta.Comparison(), score_weight=score_weight)

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class PixelOperator(Operator):
    def __init__(self, score_weight: float = 1e0):
        msg = build_deprecation_message(
            "The ABC PixelOperator",
            "0.4.0",
            info=(
                "The same behavior can be achieved by passing "
                "domain=pystiche.ops.meta.Pixel() to Operator."
            ),
        )
        warnings.warn(msg)
        super().__init__(domain=meta.Pixel(), score_weight=score_weight)

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class EncodingOperator(Operator):
    def __init__(self, score_weight: float = 1e0):
        msg = build_deprecation_message(
            "The ABC EncodingOperator",
            "0.4.0",
            info=(
                "The same behavior can be achieved by passing "
                "domain=pystiche.ops.meta.Encoding() to Operator."
            ),
        )
        warnings.warn(msg)
        super().__init__(domain=meta.Encoding(), score_weight=score_weight)

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class PixelRegularizationOperator(Operator):
    def __init__(self, score_weight: float = 1.0):
        super().__init__(
            cls=meta.Regularization(), domain=meta.Pixel(), score_weight=score_weight,
        )

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        input_repr = self.input_image_to_repr(image)
        return self.calculate_score(input_repr)

    @abstractmethod
    def input_image_to_repr(
        self, image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        pass

    @abstractmethod
    def calculate_score(
        self, input_repr: Union[torch.Tensor, pystiche.TensorStorage]
    ) -> torch.Tensor:
        pass


class EncodingRegularizationOperator(Operator):
    def __init__(self, encoder: Encoder, score_weight: float = 1.0):
        super().__init__(
            cls=meta.Regularization(),
            domain=meta.Encoding(),
            score_weight=score_weight,
        )
        self.encoder = encoder

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        input_repr = self.input_image_to_repr(image)
        return self.calculate_score(input_repr)

    def input_image_to_repr(
        self, image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        enc = self.encoder(image)
        return self.input_enc_to_repr(enc)

    @abstractmethod
    def input_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        pass

    @abstractmethod
    def calculate_score(
        self, input_repr: Union[torch.Tensor, pystiche.TensorStorage]
    ) -> torch.Tensor:
        pass

    def _properties(self):
        dct = super()._properties()
        dct["encoder"] = self.encoder
        return dct

    def __repr__(self) -> str:
        return self._build_repr(named_children=())


class PixelComparisonOperator(Operator):
    def __init__(self, score_weight: float = 1e0):
        super().__init__(
            cls=meta.Comparison(), domain=meta.Pixel(), score_weight=score_weight
        )

    def set_target_image(self, image: torch.Tensor):
        with torch.no_grad():
            repr, ctx = self.target_image_to_repr(image)
        self.register_buffer("target_image", image)
        self.register_buffer("target_repr", repr)
        self.register_buffer("ctx", ctx)

    @property
    def has_target_image(self) -> bool:
        return "target_image" in self._buffers

    @abstractmethod
    def target_image_to_repr(
        self, image: torch.Tensor
    ) -> Tuple[
        Union[torch.Tensor, pystiche.TensorStorage],
        Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ]:
        pass

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.has_target_image:
            # TODO: message
            raise RuntimeError

        target_repr, ctx = self.target_repr, self.ctx
        input_repr = self.input_image_to_repr(image, ctx)
        return self.calculate_score(input_repr, target_repr, ctx)

    @abstractmethod
    def input_image_to_repr(
        self,
        image: torch.Tensor,
        ctx: Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        pass

    @abstractmethod
    def calculate_score(
        self,
        input_repr: Union[torch.Tensor, pystiche.TensorStorage],
        target_repr: Union[torch.Tensor, pystiche.TensorStorage],
        ctx: Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ) -> torch.Tensor:
        pass


class EncodingComparisonOperator(Operator):
    def __init__(self, encoder: Encoder, score_weight: float = 1.0):
        super().__init__(
            cls=meta.Comparison(), domain=meta.Encoding(), score_weight=score_weight,
        )
        self.encoder = encoder

    def set_target_image(self, image: torch.Tensor):
        with torch.no_grad():
            repr, ctx = self.target_image_to_repr(image)
        self.register_buffer("target_image", image)
        self.register_buffer("target_repr", repr)
        self.register_buffer("ctx", ctx)

    @property
    def has_target_image(self) -> bool:
        return "target_image" in self._buffers

    def target_image_to_repr(
        self, image: torch.Tensor
    ) -> Tuple[
        Union[torch.Tensor, pystiche.TensorStorage],
        Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ]:
        enc = self.encoder(image)
        return self.target_enc_to_repr(enc)

    @abstractmethod
    def target_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Tuple[
        Union[torch.Tensor, pystiche.TensorStorage],
        Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ]:
        pass

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.has_target_image:
            # TODO: message
            raise RuntimeError

        target_repr, ctx = self.target_repr, self.ctx
        input_repr = self.input_image_to_repr(image, ctx)
        return self.calculate_score(input_repr, target_repr, ctx)

    def input_image_to_repr(
        self,
        image: torch.Tensor,
        ctx: Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        enc = self.encoder(image)
        return self.input_enc_to_repr(enc, ctx)

    @abstractmethod
    def input_enc_to_repr(
        self,
        enc: torch.Tensor,
        ctx: Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        pass

    @abstractmethod
    def calculate_score(
        self,
        input_repr: Union[torch.Tensor, pystiche.TensorStorage],
        target_repr: Union[torch.Tensor, pystiche.TensorStorage],
        ctx: Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ) -> torch.Tensor:
        pass

    def _properties(self):
        dct = super()._properties()
        dct["encoder"] = self.encoder
        return dct

    def __repr__(self) -> str:
        return self._build_repr(named_children=())
