from typing import Any, Union, Generator, Sequence, Tuple, Dict
import itertools
import numpy as np
import torch
import pystiche
from pystiche.typing import Numeric
from pystiche.misc import to_2d_arg
from pystiche.image.transforms import TransformMotifAffinely
from pystiche.enc import Encoder
from pystiche.nst import representation as R, functional as F
from ._base import EncodingComparisonOperator

__all__ = [
    "DirectEncodingComparisonOperator",
    "GramEncodingComparisonOperator",
    "MRFEncodingComparisonOperator",
]


class DirectEncodingComparisonOperator(EncodingComparisonOperator):
    def __init__(
        self,
        encoder: Encoder,
        layers: Sequence[str],
        name: str = "Direct enc comparison",
        **kwargs,
    ) -> None:
        super().__init__(encoder, layers, name, **kwargs)

    def _enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return enc

    def _input_enc_to_repr(self, enc: torch.Tensor, ctx: None) -> torch.Tensor:
        return self._enc_to_repr(enc)

    def _target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[Any, None]:
        return self._enc_to_repr(enc), None

    def _calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        return F.mse_loss(input_repr, target_repr)


class GramEncodingComparisonOperator(EncodingComparisonOperator):
    def __init__(
        self,
        encoder: Encoder,
        layers: Sequence[str],
        name: str = "Gram enc comparison",
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(encoder, layers, name, **kwargs)
        self.normalize = normalize

    def _enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return R.calculate_gram_repr(enc, self.normalize)

    def _input_enc_to_repr(self, enc: torch.Tensor, ctx: None) -> torch.Tensor:
        return self._enc_to_repr(enc)

    def _target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self._enc_to_repr(enc), None

    def _calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        return F.mse_loss(input_repr, target_repr)

    def _descriptions(self) -> Dict[str, Any]:
        dct = super()._descriptions()
        dct["Normalize Gram matrix"] = self.normalize
        return dct


class MRFEncodingComparisonOperator(EncodingComparisonOperator):
    def __init__(
        self,
        encoder: Encoder,
        layers: Sequence[str],
        patch_size: Union[int, Sequence[int]],
        name: str = "MRF enc comparison",
        stride: Union[int, Sequence[int]] = 1,
        num_scale_steps: int = 0,
        scale_step_width: Numeric = 5e-2,
        num_rotation_steps: int = 0,
        rotation_step_width: Numeric = 10,
        **kwargs,
    ):
        super().__init__(encoder, layers, name, **kwargs)
        self.patch_size = to_2d_arg(patch_size)
        self.stride = to_2d_arg(stride)
        self.num_scale_steps = num_scale_steps
        self.scale_step_width = scale_step_width
        self.num_rotation_steps = num_rotation_steps
        self.rotation_step_width = rotation_step_width

    def _enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return R.calculate_mrf_repr(enc, self.patch_size, stride=self.stride)

    def _input_enc_to_repr(self, enc: torch.Tensor, ctx: None) -> torch.Tensor:
        return self._enc_to_repr(enc)

    def _target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self._enc_to_repr(enc), None

    def _calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        return F.patch_matching_loss(
            input_repr, target_repr, reduction=self.loss_reduction
        )

    def _process_target(self, image: torch.Tensor) -> Any:
        reprs = [[] for _ in self.layers]
        for transform in self._target_transforms(image.device):
            image_encs = self.encoder(transform(image), self.layers)
            image_reprs, _ = self._target_encs_to_reprs(image_encs)

            for repr, image_repr in zip(reprs, image_reprs):
                repr.append(image_repr)

        reprs = [torch.cat(image_reprs) for image_reprs in reprs]
        ctxs = [None] * len(reprs)
        return pystiche.tuple(reprs), pystiche.tuple(ctxs)

    def _target_transforms(
        self, device: torch.device
    ) -> Generator[TransformMotifAffinely, None, None]:
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
            yield transform.to(device)

    def _descriptions(self) -> Dict[str, Any]:
        dct = super()._descriptions()
        dct["Patch size"] = self.patch_size
        dct["Stride"] = self.stride
        if self.num_scale_steps > 0:
            dct["Number of scale steps"] = self.num_scale_steps
            dct["Scale step width"] = f"{self.scale_step_width:.1%}"
        if self.num_rotation_steps > 0:
            dct["Number of rotation steps"] = self.num_rotation_steps
            dct["Rotation step width"] = f"{self.rotation_step_width:.1f}°"
        return dct
