import itertools
import warnings
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

import torch

import pystiche
from pystiche.enc import Encoder
from pystiche.image.transforms import TransformMotifAffinely
from pystiche.misc import build_deprecation_message, to_2d_arg

from . import functional as F
from .op import EncodingComparisonOperator

__all__ = [
    "FeatureReconstructionOperator",
    "MSEEncodingOperator",
    "GramOperator",
    "MRFOperator",
]


class FeatureReconstructionOperator(EncodingComparisonOperator):
    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return enc

    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc), None

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.mse_loss(input_repr, target_repr)


class MSEEncodingOperator(FeatureReconstructionOperator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        msg = build_deprecation_message(
            "The class MSEEncodingOperator",
            "0.4.0",
            info="It was renamed to FeatureReconstructionOperator.",
        )
        warnings.warn(msg)
        super().__init__(*args, **kwargs)


class GramOperator(EncodingComparisonOperator):
    def __init__(
        self, encoder: Encoder, normalize: bool = True, score_weight: float = 1.0
    ) -> None:
        super().__init__(encoder, score_weight=score_weight)
        self.normalize = normalize

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return pystiche.batch_gram_matrix(enc, normalize=self.normalize)

    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc), None

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.mse_loss(input_repr, target_repr)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        if not self.normalize:
            dct["normalize"] = self.normalize
        return dct


class MRFOperator(EncodingComparisonOperator):
    def __init__(
        self,
        encoder: Encoder,
        patch_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        num_scale_steps: int = 0,
        scale_step_width: float = 5e-2,
        num_rotation_steps: int = 0,
        rotation_step_width: float = 10,
        score_weight: float = 1.0,
    ):
        super().__init__(encoder, score_weight=score_weight)
        self.patch_size = to_2d_arg(patch_size)
        self.stride = to_2d_arg(stride)
        self.num_scale_steps = num_scale_steps
        self.scale_step_width = scale_step_width
        self.num_rotation_steps = num_rotation_steps
        self.rotation_step_width = rotation_step_width

    def set_target_guide(self, guide: torch.Tensor, recalc_repr: bool = True) -> None:
        # Since the target representation of the MRFOperator possibly comprises
        # scaled or rotated patches, it is not useful to store the target encoding
        # guides
        self.register_buffer("target_guide", guide)
        if recalc_repr and self.has_target_image:
            self.set_target_image(self.target_image)

    def _guide_repr(self, repr: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Due to the guiding large areas of the images might be zero and thus many
        # patches might carry no information. These patches can be removed from the
        # target and input representation reducing the computing cost and memory during
        # the loss calculation.

        # Patches without information have constant values in the spatial dimensions.
        repr_flat = torch.flatten(repr, 2)
        constant = repr_flat[:, :, 0].unsqueeze(2)

        # By checking where the spatial values do not differ from this constant in any
        # channel, the patches with no information can be filtered out.
        abs_diff = torch.abs(repr_flat - constant)
        mask = torch.any(torch.flatten(abs_diff > eps, 1), dim=1)

        return repr[mask]

    def enc_to_repr(self, enc: torch.Tensor, is_guided: bool) -> torch.Tensor:
        repr = pystiche.extract_patches2d(enc, self.patch_size, self.stride)
        if not is_guided:
            return repr

        return self._guide_repr(repr)

    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.enc_to_repr(enc, self.has_input_guide)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc, self.has_target_guide), None

    def target_image_to_repr(self, image: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Due to the possible scaling and rotation, we only apply the guide to the
        # target image and not the encodings
        if self.has_target_guide:
            image = self.apply_guide(image, self.target_guide)
        device = image.device
        reprs = []
        for transform in self._target_image_transforms():
            transform = transform.to(device)
            enc = self.encoder(transform(image))
            repr, _ = self.target_enc_to_repr(enc)
            reprs.append(repr)

        repr = torch.cat(reprs)
        ctx = None
        return repr, ctx

    def _target_image_transforms(self,) -> Iterator[TransformMotifAffinely]:
        scaling_factors = np.arange(
            -self.num_scale_steps, self.num_scale_steps + 1, dtype=np.float
        )
        scaling_factors = 1.0 + (scaling_factors * self.scale_step_width)

        rotation_angles = np.arange(
            -self.num_rotation_steps, self.num_rotation_steps + 1, dtype=np.float
        )
        rotation_angles *= self.rotation_step_width

        for transform_params in itertools.product(scaling_factors, rotation_angles):
            scaling_factor, rotation_angle = transform_params
            transform = TransformMotifAffinely(
                scaling_factor=scaling_factor,
                rotation_angle=rotation_angle,
                canvas="same",  # FIXME: this should be valid after it is implemented
            )
            yield transform

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.patch_matching_loss(input_repr, target_repr)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["patch_size"] = self.patch_size
        dct["stride"] = self.stride
        if self.num_scale_steps > 0:
            dct["num_scale_steps"] = self.num_scale_steps
            dct["scale_step_width"] = f"{self.scale_step_width:.1%}"
        if self.num_rotation_steps > 0:
            dct["num_rotation_steps"] = self.num_rotation_steps
            dct["rotation_step_width"] = f"{self.rotation_step_width:.1f}°"
        return dct
