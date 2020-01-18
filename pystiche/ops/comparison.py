from typing import Union, Iterator, Sequence, Tuple
import itertools
import numpy as np
import torch
import pystiche
from pystiche.misc import to_2d_arg
from pystiche.image.transforms import TransformMotifAffinely
from pystiche.enc import Encoder
from pystiche import functional as F
from .op import EncodingComparisonOperator

__all__ = ["MSEEncodingLoss", "GramLoss", "MRFLoss"]


class MSEEncodingLoss(EncodingComparisonOperator):
    def image_to_enc(self, *args, **kwargs):
        return super().image_to_enc(*args, **kwargs)

    def _enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return enc

    def input_enc_to_repr(self, enc: torch.Tensor, ctx: None) -> torch.Tensor:
        return self._enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self._enc_to_repr(enc), None

    def calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        return F.mse_loss(input_repr, target_repr)


class GramLoss(EncodingComparisonOperator):
    def __init__(
        self,
        encoder: Encoder,
        layer: str,
        normalize: bool = True,
        score_weight: float = 1.0,
    ) -> None:
        super().__init__(encoder, layer, score_weight=score_weight)
        self.normalize = normalize  # FIXME: add to description

    def _enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(enc, 2)
        G = torch.bmm(x, x.transpose(1, 2))
        if self.normalize:
            return G / x.size()[-1]
        else:
            return G

    def input_enc_to_repr(self, enc: torch.Tensor, ctx: None) -> torch.Tensor:
        return self._enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self._enc_to_repr(enc), None

    def calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        return F.mse_loss(input_repr, target_repr)


class MRFLoss(EncodingComparisonOperator):
    def __init__(
        self,
        encoder: Encoder,
        layer: str,
        patch_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        num_scale_steps: int = 0,
        scale_step_width: float = 5e-2,
        num_rotation_steps: int = 0,
        rotation_step_width: float = 10,
        score_weight: float = 1.0,
    ):
        super().__init__(encoder, layer, score_weight=score_weight)
        self.patch_size = to_2d_arg(patch_size)
        self.stride = to_2d_arg(stride)
        self.num_scale_steps = num_scale_steps
        self.scale_step_width = scale_step_width
        self.num_rotation_steps = num_rotation_steps
        self.rotation_step_width = rotation_step_width

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return pystiche.extract_patches2d(enc, self.patch_size, self.stride)

    def input_enc_to_repr(self, enc: torch.Tensor, ctx: None) -> torch.Tensor:
        return self._enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self._enc_to_repr(enc), None

    def target_image_to_repr(self, image: torch.Tensor) -> Tuple[torch.Tensor, None]:
        device = image.device
        reprs = []
        for transform in self._target_image_transforms():
            transform = transform.to(device)
            enc = self._image_to_enc(transform(image))
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
                scaling_factors=scaling_factor,
                rotation_angle=rotation_angle,
                canvas="same",  # FIXME: this should be valid after it is implemented
            )
            yield transform

    def calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        return F.patch_matching_loss(input_repr, target_repr)

    # def _descriptions(self) -> Dict[str, Any]:
    #     dct = super()._descriptions()
    #     dct["Patch size"] = self.patch_size
    #     dct["Stride"] = self.stride
    #     if self.num_scale_steps > 0:
    #         dct["Number of scale steps"] = self.num_scale_steps
    #         dct["Scale step width"] = f"{self.scale_step_width:.1%}"
    #     if self.num_rotation_steps > 0:
    #         dct["Number of rotation steps"] = self.num_rotation_steps
    #         dct["Rotation step width"] = f"{self.rotation_step_width:.1f}°"
    #     return dct
