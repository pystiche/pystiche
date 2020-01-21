from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
import torch
import pystiche
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


class Operator(ABC, pystiche.Module):
    def __init__(self, score_weight: float = 1.0) -> None:
        super().__init__()
        self.score_weight = score_weight

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.process_input_image(input_image) * self.score_weight

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


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
