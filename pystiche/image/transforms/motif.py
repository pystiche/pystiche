from typing import Any, Dict, Optional, Tuple, Union, cast

import torch

from . import functional as F
from .core import Transform

__all__ = [
    "TransformMotifAffinely",
    "ShearMotif",
    "RotateMotif",
    "ScaleMotif",
    "TranslateMotif",
]


class TransformMotifAffinely(Transform):
    def __init__(
        self,
        shearing_angle: Optional[float] = None,
        clockwise_shearing: bool = False,
        shearing_center: Optional[Tuple[float, float]] = None,
        rotation_angle: Optional[float] = None,
        clockwise_rotation: bool = False,
        rotation_center: Optional[Tuple[float, float]] = None,
        scaling_factor: Optional[Union[float, Tuple[float, float]]] = None,
        scaling_center: Optional[Tuple[float, float]] = None,
        translation: Optional[Tuple[float, float]] = None,
        inverse_translation: bool = False,
        canvas: str = "same",
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.shearing_angle = shearing_angle
        self.clockwise_shearing = clockwise_shearing
        self.shearing_center = shearing_center
        self.rotation_angle = rotation_angle
        self.clockwise_rotation = clockwise_rotation
        self.rotation_center = rotation_center
        self.scaling_factor = scaling_factor
        self.scaling_center = scaling_center
        self.translation = translation
        self.inverse_translation = inverse_translation
        self.canvas = canvas
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode

        if not self.has_affine_params:
            msg = (
                "TransformMotifAffinely was created without any affine parameter. At "
                "least one of shearing_angle, rotation_angle, scaling_factor, and "
                "translation has to be set."
            )
            raise RuntimeError(msg)

    @property
    def has_shearing(self) -> bool:
        return self.shearing_angle is not None

    @property
    def has_rotation(self) -> bool:
        return self.rotation_angle is not None

    @property
    def has_scaling(self) -> bool:
        return self.scaling_factor is not None

    @property
    def has_translation(self) -> bool:
        return self.translation is not None

    @property
    def has_affine_params(self) -> bool:
        return any(
            (
                self.has_rotation,
                self.has_shearing,
                self.has_scaling,
                self.has_translation,
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            F.transform_motif_affinely(
                image,
                shearing_angle=self.shearing_angle,
                clockwise_shearing=self.clockwise_shearing,
                shearing_center=self.shearing_center,
                rotation_angle=self.rotation_angle,
                clockwise_rotation=self.clockwise_rotation,
                rotation_center=self.rotation_center,
                scaling_factor=self.scaling_factor,
                scaling_center=self.scaling_center,
                translation=self.translation,
                inverse_translation=self.inverse_translation,
                canvas=self.canvas,
                interpolation_mode=self.interpolation_mode,
                padding_mode=self.padding_mode,
            ),
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        if self.has_shearing:
            dct["shearing_angle"] = f"{self.shearing_angle}°"
            if self.clockwise_shearing:
                dct["clockwise_shearing"] = self.clockwise_shearing
            if self.shearing_center is not None:
                dct["shearing_center"] = self.shearing_center
        if self.has_rotation:
            dct["rotation_angle"] = f"{self.rotation_angle}°"
            if self.clockwise_rotation:
                dct["clockwise_rotation"] = self.clockwise_rotation
            if self.rotation_center is not None:
                dct["rotation_center"] = self.rotation_center
        if self.has_scaling:
            dct["scaling_factor"] = self.scaling_factor
            if self.scaling_center is not None:
                dct["scaling_center"] = self.scaling_center
        if self.has_translation:
            dct["translation"] = self.translation
            if self.inverse_translation:
                dct["inverse_translation"] = self.inverse_translation
        if self.canvas != "same":
            dct["canvas"] = self.canvas
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        if self.padding_mode != "zeros":
            dct["padding_mode"] = self.padding_mode
        return dct


class ShearMotif(Transform):
    def __init__(
        self,
        angle: float,
        clockwise: bool = False,
        center: Optional[Tuple[float, float]] = None,
        canvas: str = "same",
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.angle = angle
        self.clockwise = clockwise
        self.center = center
        self.canvas = canvas
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.shear_motif(
            image,
            self.angle,
            clockwise=self.clockwise,
            center=self.center,
            canvas=self.canvas,
            interpolation_mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["angle"] = f"{self.angle}°"
        if self.clockwise:
            dct["clockwise"] = self.clockwise
        if self.center is not None:
            dct["center"] = self.center
        if self.canvas != "same":
            dct["canvas"] = self.canvas
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        if self.padding_mode != "zeros":
            dct["padding_mode"] = self.padding_mode
        return dct


class RotateMotif(Transform):
    def __init__(
        self,
        angle: float,
        clockwise: bool = False,
        center: Optional[Tuple[float, float]] = None,
        canvas: str = "same",
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.angle = angle
        self.clockwise = clockwise
        self.center = center
        self.canvas = canvas
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.rotate_motif(
            image,
            self.angle,
            clockwise=self.clockwise,
            center=self.center,
            canvas=self.canvas,
            interpolation_mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["angle"] = f"{self.angle}°"
        if self.clockwise:
            dct["clockwise"] = self.clockwise
        if self.center is not None:
            dct["center"] = self.center
        if self.canvas != "same":
            dct["canvas"] = self.canvas
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        if self.padding_mode != "zeros":
            dct["padding_mode"] = self.padding_mode
        return dct


class ScaleMotif(Transform):
    def __init__(
        self,
        factor: Union[float, Tuple[float, float]],
        center: Optional[Tuple[float, float]] = None,
        canvas: str = "same",
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.factor = factor
        self.center = center
        self.canvas = canvas
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.scale_motif(
            image,
            self.factor,
            center=self.center,
            canvas=self.canvas,
            interpolation_mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["factor"] = self.factor
        if self.center is not None:
            dct["center"] = self.center
        if self.canvas != "same":
            dct["canvas"] = self.canvas
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        if self.padding_mode != "zeros":
            dct["padding_mode"] = self.padding_mode
        return dct


class TranslateMotif(Transform):
    def __init__(
        self,
        translation: Tuple[float, float],
        inverse: bool = False,
        canvas: str = "same",
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.translation = translation
        self.inverse = inverse
        self.canvas = canvas
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.translate_motif(
            image,
            self.translation,
            inverse=self.inverse,
            canvas=self.canvas,
            interpolation_mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["translation"] = self.translation
        if self.inverse:
            dct["inverse"] = self.inverse
        if self.canvas != "same":
            dct["canvas"] = self.canvas
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        if self.padding_mode != "zeros":
            dct["padding_mode"] = self.padding_mode
        return dct
