from .op import (
    PixelRegularizationOperator,
    EncodingRegularizationOperator,
    PixelComparisonOperator,
    EncodingComparisonOperator,
)
from typing import Union, Optional, Tuple
import torch
import pystiche

__all__ = [
    "Guidance",
    "RegularizationGuidance",
    "ComparisonGuidance",
    "PixelGuidance",
    "EncodingGuidance",
    "PixelRegularizationGuidance",
    "EncodingRegularizationGuidance",
    "PixelComparisonGuidance",
    "EncodingComparisonGuidance",
]


class Guidance:
    @staticmethod
    def apply_guide(image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        return image * guide


class RegularizationGuidance(Guidance):
    pass


class ComparisonGuidance(Guidance):
    pass


class PixelGuidance(Guidance):
    pass


class EncodingGuidance(Guidance):
    pass


class PixelRegularizationGuidance(PixelGuidance, RegularizationGuidance):
    def __init__(self, *args, **kwargs):
        if not isinstance(self, PixelRegularizationOperator):
            raise RuntimeError
        super().__init__(*args, **kwargs)

    def set_input_guide(self, guide: torch.Tensor) -> None:
        self.register_buffer("input_guide", guide)

    @property
    def has_input_guide(self) -> bool:
        return self.input_guide is not None

    def input_image_to_repr(
        self, image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        guided_image = self.apply_guide(image, self.input_guide)
        return super().input_image_to_repr(guided_image)


class EncodingRegularizationGuidance(EncodingGuidance, RegularizationGuidance):
    def __init__(self, *args, **kwargs):
        if not isinstance(self, EncodingRegularizationOperator):
            raise RuntimeError
        super().__init__(*args, **kwargs)

    def set_input_guide(self, guide: torch.Tensor) -> None:
        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer("input_guide", guide)
        self.register_buffer("input_enc_guide", enc_guide)

    @property
    def has_input_guide(self) -> bool:
        return self.input_guide is not None

    def input_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        guided_enc = self.apply_guide(enc, self.input_enc_guide)
        return super().input_enc_to_repr(guided_enc)


class PixelComparisonGuidance(PixelGuidance, ComparisonGuidance):
    def __init__(self, *args, **kwargs):
        if not isinstance(self, PixelComparisonOperator):
            raise RuntimeError
        super().__init__(*args, **kwargs)

    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        self.register_buffer("target_guide", guide)
        if recalc_repr and self.has_target_image:
            self.set_target_image(self.target_image)

    @property
    def has_target_guide(self) -> bool:
        return "target_guide" in self._buffers

    def target_image_to_repr(
        self, image: torch.Tensor
    ) -> Tuple[
        Union[torch.Tensor, pystiche.TensorStorage],
        Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ]:
        guided_image = self.apply_guide(image, self.target_guide)
        return super().target_image_to_repr(guided_image)

    def set_input_guide(self, guide: torch.Tensor) -> None:
        self.register_buffer("input_guide", guide)

    @property
    def has_input_guide(self) -> bool:
        return "input_guide" in self._buffers

    def input_image_to_repr(
        self,
        image: torch.Tensor,
        ctx: Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        guided_image = self.apply_guide(image, self.target_guide)
        return super().input_image_to_repr(guided_image, ctx)


class EncodingComparisonGuidance(EncodingGuidance, ComparisonGuidance):
    def __init__(self, *args, **kwargs):
        if not isinstance(self, EncodingComparisonOperator):
            raise RuntimeError
        super().__init__(*args, **kwargs)

    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer("target_guide", guide)
        self.register_buffer("target_enc_guide", enc_guide)
        if recalc_repr and self.has_target_image:
            self.set_target_image(self.target_image)

    @property
    def has_target_guide(self) -> bool:
        return "target_guide" in self._buffers

    def target_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Tuple[
        Union[torch.Tensor, pystiche.TensorStorage],
        Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ]:
        guided_enc = self.apply_guide(enc, self.target_enc_guide)
        return super().target_enc_to_repr(guided_enc)

    def set_input_guide(self, guide: torch.Tensor) -> None:
        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer("input_guide", guide)
        self.register_buffer("input_enc_guide", enc_guide)

    @property
    def has_input_guide(self) -> bool:
        return "input_guide" in self._buffers

    def input_enc_to_repr(
        self,
        enc: torch.Tensor,
        ctx: Optional[Union[torch.Tensor, pystiche.TensorStorage]],
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        guided_enc = self.apply_guide(enc, self.input_enc_guide)
        return super().input_enc_to_repr(guided_enc, ctx)
