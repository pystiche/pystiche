from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch

import pystiche
from pystiche.enc import Encoder
from pystiche.misc import is_almost, to_engstr

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
    def __init__(self, score_weight: float = 1e0,) -> None:
        super().__init__()
        self.score_weight = score_weight

    input_guide: torch.Tensor

    def set_input_guide(self, guide: torch.Tensor) -> None:
        self.register_buffer("input_guide", guide)

    @property
    def has_input_guide(self) -> bool:
        return "input_guide" in self._buffers

    @staticmethod
    def apply_guide(image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        return image * guide

    def forward(
        self, input_image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.LossDict]:
        return self.process_input_image(input_image) * self.score_weight

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def named_operators(
        self, recurse: bool = False,
    ) -> Iterator[Tuple[str, "Operator"]]:
        if recurse:
            iterator = self.named_modules()
            # First module is always self so dismiss that
            next(iterator)
        else:
            iterator = self.named_children()
        for name, child in iterator:
            if isinstance(child, Operator):
                yield name, child

    def operators(self, recurse: bool = False) -> Iterator["Operator"]:
        for _, op in self.named_operators(recurse=recurse):
            yield op

    def _properties(self) -> Dict[str, Any]:
        dct = OrderedDict()
        if not is_almost(self.score_weight, 1e0):
            dct["score_weight"] = to_engstr(self.score_weight)
        return dct


class RegularizationOperator(Operator):
    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class ComparisonOperator(Operator):
    target_guide: torch.Tensor
    target_image: torch.Tensor
    target_repr: torch.Tensor
    ctx: Optional[torch.Tensor]

    @abstractmethod
    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        pass

    @property
    def has_target_guide(self) -> bool:
        return "target_guide" in self._buffers

    @abstractmethod
    def set_target_image(self, image: torch.Tensor) -> None:
        pass

    @property
    def has_target_image(self) -> bool:
        return "target_image" in self._buffers

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class PixelOperator(Operator):
    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass


class EncodingOperator(Operator):
    @property
    @abstractmethod
    def encoder(self) -> Encoder:
        pass

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["encoder"] = self.encoder
        return dct

    def __repr__(self) -> str:
        return self._build_repr(
            named_children=[
                (name, child)
                for name, child in self.named_children()
                if child is not self.encoder
            ]
        )


class PixelRegularizationOperator(PixelOperator, RegularizationOperator):
    def __init__(self, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight,)

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.has_input_guide:
            image = self.apply_guide(image, self.input_guide)
        input_repr = self.input_image_to_repr(image)
        return self.calculate_score(input_repr)

    @abstractmethod
    def input_image_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        pass


class EncodingRegularizationOperator(EncodingOperator, RegularizationOperator):
    def __init__(self, encoder: Encoder, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight)
        self._encoder = encoder

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    def set_input_guide(self, guide: torch.Tensor) -> None:
        super().set_input_guide(guide)
        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer("input_enc_guide", enc_guide)

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        input_repr = self.input_image_to_repr(image)
        return self.calculate_score(input_repr)

    def input_image_to_repr(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(image)
        if self.has_input_guide:
            enc = self.apply_guide(enc, self.input_guide)
        return self.input_enc_to_repr(enc)

    @abstractmethod
    def input_enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        pass


class PixelComparisonOperator(PixelOperator, ComparisonOperator):
    def __init__(self, score_weight: float = 1e0):
        super().__init__(score_weight=score_weight)

    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        self.register_buffer("target_guide", guide)
        if recalc_repr and self.has_target_image:
            self.set_target_image(self.target_image)

    def set_target_image(self, image: torch.Tensor) -> None:
        self.register_buffer("target_image", image)
        with torch.no_grad():
            if self.has_target_guide:
                image = self.apply_guide(image, self.target_guide)
            repr, ctx = self.target_image_to_repr(image)
        self.register_buffer("target_repr", repr)
        if ctx is not None:
            self.register_buffer("ctx", ctx)
        else:
            self.ctx = None

    @abstractmethod
    def target_image_to_repr(
        self, image: torch.Tensor
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor],
    ]:
        pass

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.has_target_image:
            # TODO: message
            raise RuntimeError
        target_repr, ctx = self.target_repr, self.ctx

        if self.has_input_guide:
            image = self.apply_guide(image, self.input_guide)
        input_repr = self.input_image_to_repr(image, ctx)

        return self.calculate_score(input_repr, target_repr, ctx)

    @abstractmethod
    def input_image_to_repr(
        self, image: torch.Tensor, ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pass


class EncodingComparisonOperator(EncodingOperator, ComparisonOperator):
    target_enc_guide: torch.Tensor
    input_enc_guide: torch.Tensor

    def __init__(self, encoder: Encoder, score_weight: float = 1.0):
        super().__init__(score_weight=score_weight)
        self._encoder = encoder

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer("target_guide", guide)
        self.register_buffer("target_enc_guide", enc_guide)
        if recalc_repr and self.has_target_image:
            self.set_target_image(self.target_image)

    def set_target_image(self, image: torch.Tensor) -> None:
        with torch.no_grad():
            repr, ctx = self.target_image_to_repr(image)
        self.register_buffer("target_image", image)
        self.register_buffer("target_repr", repr)
        if ctx is not None:
            self.register_buffer("ctx", ctx)
        else:
            self.ctx = None

    def target_image_to_repr(
        self, image: torch.Tensor
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor],
    ]:
        enc = self.encoder(image)
        if self.has_target_guide:
            enc = self.apply_guide(enc, self.target_enc_guide)
        return self.target_enc_to_repr(enc)

    @abstractmethod
    def target_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor],
    ]:
        pass

    def set_input_guide(self, guide: torch.Tensor) -> None:
        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer("input_guide", guide)
        self.register_buffer("input_enc_guide", enc_guide)

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.has_target_image:
            # TODO: message
            raise RuntimeError
        target_repr, ctx = self.target_repr, self.ctx
        input_repr = self.input_image_to_repr(image, ctx)
        return self.calculate_score(input_repr, target_repr, ctx)

    def input_image_to_repr(
        self, image: torch.Tensor, ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        enc = self.encoder(image)
        if self.has_input_guide:
            enc = self.apply_guide(enc, self.input_enc_guide)
        return self.input_enc_to_repr(enc, ctx)

    @abstractmethod
    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pass
