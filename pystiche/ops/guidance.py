from typing import Union
import torch
import pystiche


class Guidance(pystiche.StateObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_guide = None

    def set_input_guide(self, guide: torch.Tensor) -> None:
        self.input_guide = guide.detach()

    @property
    def has_input_guide(self) -> bool:
        return self.input_guide is not None

    @staticmethod
    def apply_guide(image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        return image * guide


class RegularizationGuidance(Guidance):
    def _input_image_to_repr(
        self, image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.StateObject]:
        guided_image = self.apply_guide(image, self.input_guide)
        return super()._input_image_to_repr(guided_image)


class EncodingRegularizationGuidance(Guidance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_enc_guide = None

    def _propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        return self.encoder.propagate_guide(guide, (self.layer,))[0]

    def set_input_guide(self, guide: torch.Tensor):
        super().set_input_guide(guide)
        with torch.no_grad:
            input_enc_guide = self.encoder.propagate_guide(guide, ())
        self._input_enc_guide = input_enc_guide.detach()

    def _input_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.StateObject]:
        guided_enc = self.apply_guide(enc, self._input_enc_guide)
        return super()._input_enc_to_repr(guided_enc)
