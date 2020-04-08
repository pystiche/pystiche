import itertools
from typing import Iterator, Sequence, Tuple, Union

import numpy as np

import torch

import pystiche
from pystiche.enc import Encoder
from pystiche.image.transforms import TransformMotifAffinely
from pystiche.misc import to_2d_arg

from . import functional as F
from .op import EncodingComparisonOperator

__all__ = ["MSEEncodingOperator", "GramOperator", "MRFOperator"]


class MSEEncodingOperator(EncodingComparisonOperator):
    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return enc

    def input_enc_to_repr(self, enc: torch.Tensor, ctx: None) -> torch.Tensor:
        return self.enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc), None

    def calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        return F.mse_loss(input_repr, target_repr)


class GramOperator(EncodingComparisonOperator):
    def __init__(
        self, encoder: Encoder, normalize: bool = True, score_weight: float = 1.0
    ) -> None:
        super().__init__(encoder, score_weight=score_weight)
        self.normalize = normalize

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return pystiche.batch_gram_matrix(enc, normalize=self.normalize)

    def input_enc_to_repr(self, enc: torch.Tensor, ctx: None) -> torch.Tensor:
        return self.enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc), None

    def calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        return F.mse_loss(input_repr, target_repr)

    def _properties(self):
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

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return pystiche.extract_patches2d(enc, self.patch_size, self.stride)

    def input_enc_to_repr(self, enc: torch.Tensor, ctx: None) -> torch.Tensor:
        return self.enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc), None

    def target_image_to_repr(self, image: torch.Tensor) -> Tuple[torch.Tensor, None]:
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
        scaling_factors *= self.scale_step_width
        scaling_factors += 1.0

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
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        return F.patch_matching_loss(input_repr, target_repr)

    def _properties(self):
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
