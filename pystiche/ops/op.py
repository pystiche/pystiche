from abc import abstractmethod
from typing import Any, Union, Optional, Tuple, Dict
from collections import OrderedDict
import torch
import pystiche
from pystiche.misc import to_engstr
from pystiche.enc import Encoder

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
    def __init__(self, score_weight: float = 1e0) -> None:
        super().__init__()
        self.score_weight = score_weight

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.process_input_image(input_image) * self.score_weight

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def _properties(self) -> Dict[str, Any]:
        dct = OrderedDict()
        if abs(self.score_weight - 1e0) > 1e-6:
            dct["score_weight"] = to_engstr(self.score_weight)
        return dct


class RegularizationOperator(Operator):
    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class ComparisonOperator(Operator):
    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class PixelOperator(Operator):
    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class EncodingOperator(Operator):
    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class PixelRegularizationOperator(PixelOperator, RegularizationOperator):
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


class EncodingRegularizationOperator(EncodingOperator, RegularizationOperator):
    def __init__(self, encoder: Encoder, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight)
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

    def __str__(self) -> str:
        return self._build_str(named_children=())


class PixelComparisonOperator(PixelOperator, ComparisonOperator):
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


class EncodingComparisonOperator(EncodingOperator, ComparisonOperator):
    def __init__(self, encoder: Encoder, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight)
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

    def __str__(self) -> str:
        return self._build_str(named_children=())
