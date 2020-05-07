# type: ignore
from typing import Union, Optional, Tuple
import torch
import pystiche
from .op import (
    PixelRegularizationOperator,
    EncodingRegularizationOperator,
    PixelComparisonOperator,
    EncodingComparisonOperator,
)
from .comparison import MRFOperator


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
    "MRFOperatorGuidance",
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


class MRFOperatorGuidance(EncodingComparisonGuidance):
    def __init__(self, *args, **kwargs):
        if not isinstance(self, MRFOperator):
            raise RuntimeError
        super().__init__(*args, **kwargs)

    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        # Since the target representation of the MRFOperator possibly comprises
        # scaled or rotated patches, it is not useful to store propagated guides
        self.register_buffer("target_guide", guide)
        if recalc_repr and self.has_target_image:
            self.set_target_image(self.target_image)

    def target_image_to_repr(self, image: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Due to the possible scaling and rotation, we only apply the guide to the
        # target image and not the encodings
        if self.has_target_guide:
            image = self.apply_guide(image, self.target_guide)
        return super().target_image_to_repr(image)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # This simply overrides Guidance.target_enc_to_repr() to suppress the guiding
        # of the encodings
        return self.enc_to_repr(enc), None

    def enc_to_repr(self, enc: torch.Tensor, eps=1e-8) -> torch.Tensor:
        # Due to the guiding large areas of the images are zero and thus many patches
        # carry no information. Thus, these patches can be removed from the target and
        # input representation reducing the computing cost and memory during loss
        # calculation
        repr = super().enc_to_repr(enc)

        # Convolution operations on zero images result in patches with constant values
        # in the spatial dimensions.
        repr_flat = torch.flatten(repr, 2)
        constant = repr_flat[:, :, 0].unsqueeze(2)

        # By checking where the spatial values do not differ from this constant in any
        # channel, the patches with no information can be filtered out
        abs_diff = torch.abs(repr_flat - constant)
        mask = torch.any(torch.flatten(abs_diff > eps, 1), dim=1)

        return repr[mask]
