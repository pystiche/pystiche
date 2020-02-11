from abc import abstractmethod
from typing import Optional, Union, Iterable, Sequence, Tuple
from copy import copy
import itertools
from PIL import Image
import torch
from torch import nn
from pystiche.typing import Numeric
from pystiche.misc import to_2d_arg, to_engstr, to_engtuplestr
from .. import utils as U
from . import functional as F

__all__ = [
    "Transform",
    "Compose",
    "ImportFromPIL",
    "ExportToPIL",
    "FloatToUint8Range",
    "Uint8ToFloatRange",
    "ReverseChannelOrder",
    "Normalize",
    "Denormalize",
    "ResizeTransform",
    "Resize",
    "FixedAspectRatioResize",
    "Rescale",
    "TransformMotifAffinely",
    "ShearMotif",
    "RotateMotif",
    "ScaleMotif",
    "TranslateMotif",
    "RGBToGrayscale",
    "GrayscaleToFakegrayscale",
    "RGBToFakegrayscale",
    "GrayscaleToBinary",
    "RGBToBinary",
    "RGBToYUV",
    "YUVToRGB",
]


class Transform(nn.Module):
    @abstractmethod
    def forward(self, *input):
        pass

    def __add__(self, other: Union["Transform", "Compose"]) -> "Compose":
        return _compose_transforms(self, other)


class Compose(nn.Sequential):
    def __add__(self, other: Union["Transform", "Compose"]) -> "Compose":
        return _compose_transforms(self, other)


def _compose_transforms(*transforms: Tuple[Union[Transform, Compose], ...]) -> Compose:
    def unroll(
        transform: Union[Transform, Compose]
    ) -> Iterable[Union[Transform, Compose]]:
        if isinstance(transform, Transform):
            return (transform,)
        elif isinstance(transform, Compose):
            return transform.children()
        else:
            raise RuntimeError

    return Compose(*itertools.chain(*map(unroll, transforms)))


class ImportFromPIL(Transform):
    def __init__(
        self, device: Optional[torch.device] = None, make_batched: bool = True
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.add_batch_dim = make_batched

    def forward(self, x: Image.Image) -> torch.Tensor:
        return F.import_from_pil(x, self.device, make_batched=self.add_batch_dim)


class ExportToPIL(Transform):
    def __init__(self, mode: Optional[str] = None):
        super().__init__()
        self.mode: Optional[str] = mode

    def forward(self, x: torch.Tensor) -> Union[Image.Image, Tuple[Image.Image, ...]]:
        return F.export_to_pil(x, mode=self.mode)

    def extra_repr(self) -> str:
        return "mode={mode}".format(**self.__dict__)


class FloatToUint8Range(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.float_to_uint8_range(x)


class Uint8ToFloatRange(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.uint8_to_float_range(x)


class ReverseChannelOrder(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.reverse_channel_order(x)


class Normalize(Transform):
    def __init__(self, mean: Sequence[Numeric], std: Sequence[Numeric]) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

        self.register_buffer("_mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("_std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, self._mean, self._std)

    def extra_repr(self) -> str:
        mean = to_engtuplestr(self.mean)
        std = to_engtuplestr(self.std)
        return f"mean={mean}, std={std}"


class Denormalize(Normalize):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.denormalize(x, self._mean, self._std)


class ResizeTransform(Transform):
    def __init__(self, interpolation_mode: str = "bilinear") -> None:
        super().__init__()
        self.interpolation_mode: str = interpolation_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_size = self.calculate_image_size(x)
        return F.resize(x, image_size, interpolation_mode=self.interpolation_mode)

    @abstractmethod
    def calculate_image_size(self, x: torch.Tensor) -> Tuple[int, int]:
        pass

    @property
    @abstractmethod
    def has_fixed_size(self) -> bool:
        pass

    def extra_repr(self) -> str:
        extras = []
        resize_transform_extras = self.extra_resize_transform_repr()
        if resize_transform_extras:
            extras.append(resize_transform_extras)
        if self.interpolation_mode != "bilinear":
            extras.append(", interpolation_mode={interpolation_mode}")
        return ", ".join(extras).format(**self.__dict__)

    def extra_resize_transform_repr(self) -> str:
        return ""


class Resize(ResizeTransform):
    def __init__(self, image_size: Tuple[int, int], **kwargs):
        super().__init__(**kwargs)
        self.image_size: Tuple[int, int] = image_size

    def calculate_image_size(self, x: torch.Tensor) -> Tuple[int, int]:
        return self.image_size

    @property
    def has_fixed_size(self) -> bool:
        return True

    def extra_resize_transform_repr(self):
        return "image_size={image_size}".format(**self.__dict__)


class FixedAspectRatioResize(ResizeTransform):
    def __init__(
        self,
        edge_size: int,
        edge: str = "short",
        aspect_ratio: Optional[Numeric] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edge_size: int = edge_size
        self.edge: str = edge

        self.aspect_ratio: Optional[Numeric] = aspect_ratio
        if aspect_ratio is not None:
            self.image_size = U.edge_to_image_size(edge_size, aspect_ratio, edge)
        else:
            self.image_size = None

    def calculate_image_size(self, x: torch.Tensor) -> Tuple[int, int]:
        if self.has_fixed_size:
            return self.image_size
        else:
            aspect_ratio = U.extract_aspect_ratio(x)
            return U.edge_to_image_size(self.edge_size, aspect_ratio, self.edge)

    @property
    def has_fixed_size(self) -> bool:
        return self.image_size is not None

    def extra_resize_transform_repr(self) -> str:
        if self.has_fixed_size:
            return "size={size}".format(**self.__dict__)
        else:
            dct = copy(self.__dict__)
            dct["aspect_ratio"] = to_engstr(dct["aspect_ratio"])
            extras = (
                "edge_size={edge_size}",
                "aspect_ratio={aspect_ratio}",
                "edge={edge}",
            )
            return ", ".join(extras).format(**dct)


class Rescale(ResizeTransform):
    def __init__(self, factor: Numeric, **kwargs):
        super().__init__(**kwargs)
        self.factor: Numeric = factor

    def calculate_image_size(self, x):
        return [round(edge_size * self.factor) for edge_size in U.extract_image_size(x)]

    @property
    def has_fixed_size(self) -> bool:
        return False

    def extra_resize_transform_repr(self) -> str:
        return "factor={factor}".format(**self.__dict__)


class GridSampleTransform(Transform):
    def __init__(
        self, interpolation_mode: str = "bilinear", padding_mode: str = "zeros"
    ) -> None:
        super().__init__()
        self.interpolation_mode: str = interpolation_mode
        self.padding_mode: str = padding_mode

    @abstractmethod
    def forward(self, *input):
        pass

    def extra_repr(self) -> str:
        extras = [self.extra_grid_sample_repr()]
        if self.interpolation_mode != "bilinear":
            extras.append("interpolation_mode={interpolation_mode}")
        if self.padding_mode != "zeros":
            extras.append("padding_mode={padding_mode}")
        return ", ".join(extras).format(**self.__dict__)

    def extra_grid_sample_repr(self) -> str:
        return ""


class AffineTransform(GridSampleTransform):
    def __init__(
        self,
        image_size: Optional[Tuple[int, int]] = None,
        canvas: str = "same",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if image_size is not None:
            self.input_image_size = image_size
            transformation_matrix = self.create_transformation_matrix(image_size)
            transformation_matrix, output_image_size = F.resize_canvas(
                transformation_matrix, image_size, method=canvas
            )
            self.transformation_matrix = transformation_matrix
            self.output_image_size = output_image_size
        else:
            self.input_image_size = self.output_image_size = None
            self.transformation_matrix = None
        self.canvas = canvas

    @abstractmethod
    def create_transformation_matrix(self, image_size: Tuple[int, int]) -> torch.Tensor:
        pass

    @property
    def has_fixed_transformation_matrix(self) -> bool:
        return self.transformation_matrix is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_fixed_transformation_matrix:
            output_image_size = self.output_image_size
            transformation_matrix = self.transformation_matrix
        else:
            input_image_size = U.extract_image_size(x)
            transformation_matrix = self.create_transformation_matrix(input_image_size)
            transformation_matrix, output_image_size = F.resize_canvas(
                transformation_matrix, input_image_size, method=self.canvas
            )

        return F.transform_motif_affinely(
            x,
            transformation_matrix,
            output_image_size=output_image_size,
            interpolation_mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
        )

    def extra_grid_sample_repr(self) -> str:
        extras = [self.extra_affine_transform_repr()]
        if self.has_fixed_transformation_matrix:
            if self.input_image_size == self.output_image_size:
                extras.append("image_size={input_image_size}")
            else:
                extras.append("input_image_size={input_image_size}")
                extras.append("output_image_size={output_image_size}")
        if self.canvas != "same":
            extras.append("canvas={canvas}")
        return ", ".join(extras).format(**self.__dict__)

    def extra_affine_transform_repr(self) -> str:
        return ""


class TransformMotifAffinely(AffineTransform):
    def __init__(
        self,
        shearing_angle: Optional[Numeric] = None,
        clockwise_shearing: bool = False,
        shearing_center: Optional[Tuple[Numeric, Numeric]] = None,
        rotation_angle: Optional[Numeric] = None,
        clockwise_rotation: bool = False,
        rotation_center: Optional[Tuple[Numeric, Numeric]] = None,
        scaling_factors: Optional[Tuple[Numeric, Numeric]] = None,
        scaling_center: Optional[Tuple[Numeric, Numeric]] = None,
        translation: Optional[Tuple[Numeric, Numeric]] = None,
        inverse_translation: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.shearing_angle = shearing_angle
        self.clockwise_shearing = clockwise_shearing
        self.shearing_center = to_2d_arg(shearing_center)
        self.rotation_angle = rotation_angle
        self.clockwise_rotation = clockwise_rotation
        self.rotation_center = to_2d_arg(rotation_center)
        self.scaling_factors = to_2d_arg(scaling_factors)
        self.scaling_center = to_2d_arg(scaling_center)
        self.translation = to_2d_arg(translation)
        self.inverse_translation = inverse_translation

        if not any(
            (
                self.has_rotation,
                self.has_shearing,
                self.has_scaling,
                self.has_translation,
            )
        ):
            raise RuntimeError

    @property
    def has_shearing(self) -> bool:
        return self.shearing_angle is not None

    @property
    def has_rotation(self) -> bool:
        return self.rotation_angle is not None

    @property
    def has_scaling(self) -> bool:
        return self.scaling_factors is not None

    @property
    def has_translation(self) -> bool:
        return self.translation is not None

    def create_transformation_matrix(self, image_size: Tuple[int, int]) -> torch.Tensor:
        return F.create_affine_transformation_matrix(
            image_size,
            shearing_angle=self.shearing_angle,
            clockwise_shearing=self.clockwise_shearing,
            shearing_center=self.shearing_center,
            rotation_angle=self.rotation_angle,
            clockwise_rotation=self.clockwise_rotation,
            rotation_center=self.rotation_center,
            scaling_factors=self.scaling_factors,
            scaling_center=self.scaling_center,
            translation=self.translation,
            inverse_translation=self.inverse_translation,
        )

    def extra_affine_transform_repr(self) -> str:
        extras = []
        if self.has_shearing:
            extras.append("shearing_angle={shearing_angle}°")
            if self.clockwise_shearing:
                extras.append("clockwise_shearing={clockwise_shearing}")
            if self.shearing_center is not None:
                extras.append("shearing_center={shearing_center}")
        if self.has_rotation:
            extras.append("rotation_angle={rotation_angle}°")
            if self.clockwise_rotation:
                extras.append("clockwise_rotation={clockwise_rotation}")
            if self.rotation_center is not None:
                extras.append("rotation_center={rotation_center}")
        if self.has_scaling:
            extras.append("scaling_factors={scaling_factors}")
            if self.scaling_center is not None:
                extras.append("scaling_center={scaling_center}")
        if self.has_translation:
            extras.append("translation={translation}")
            if self.inverse_translation:
                extras.append("inverse_translation={inverse_translation}")
        return ", ".join(extras).format(**self.__dict__)


class ShearMotif(AffineTransform):
    def __init__(
        self,
        angle: Numeric,
        clockwise: bool = False,
        center: Optional[Tuple[Numeric, Numeric]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.angle = angle
        self.clockwise = clockwise
        self.center = to_2d_arg(center)

    def create_transformation_matrix(self, image_size: Tuple[int, int]) -> torch.Tensor:
        return F.create_affine_transformation_matrix(
            image_size,
            shearing_angle=self.angle,
            clockwise_shearing=self.clockwise,
            shearing_center=self.center,
        )

    def extra_affine_transform_repr(self) -> str:
        extras = ["angle={angle}°"]
        if self.clockwise:
            extras.append("clockwise={clockwise}")
        if self.center is not None:
            extras.append("center={center}")
        return ", ".join(extras).format(**self.__dict__)


class RotateMotif(AffineTransform):
    def __init__(
        self,
        angle: Numeric,
        clockwise: bool = False,
        center: Optional[Tuple[Numeric, Numeric]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.angle = angle
        self.clockwise = clockwise
        self.center = to_2d_arg(center)

    def create_transformation_matrix(self, image_size: Tuple[int, int]) -> torch.Tensor:
        return F.create_affine_transformation_matrix(
            image_size,
            rotation_angle=self.angle,
            clockwise_rotation=self.clockwise,
            rotation_center=self.center,
        )

    def extra_affine_transform_repr(self) -> str:
        extras = ["angle={angle}°"]
        if self.clockwise:
            extras.append("clockwise={clockwise}")
        if self.center is not None:
            extras.append("center={center}")
        return ", ".join(extras).format(**self.__dict__)


class ScaleMotif(AffineTransform):
    def __init__(
        self,
        factors: Union[Numeric, Tuple[Numeric, Numeric]],
        center: Optional[Tuple[Numeric, Numeric]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.factors = to_2d_arg(factors)
        self.center = to_2d_arg(center)

    def create_transformation_matrix(self, image_size: Tuple[int, int]) -> torch.Tensor:
        return F.create_affine_transformation_matrix(
            image_size, scaling_factors=self.factors, scaling_center=self.center
        )

    def extra_affine_transform_repr(self) -> str:
        extras = ["factors={factors}"]
        if self.center is not None:
            extras.append("center={center}")
        return ", ".join(extras).format(**self.__dict__)


class TranslateMotif(AffineTransform):
    def __init__(
        self,
        translation: Union[Numeric, Tuple[Numeric, Numeric]],
        inverse: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.translation = to_2d_arg(translation)
        self.inverse = inverse

    def create_transformation_matrix(self, image_size: Tuple[int, int]) -> torch.Tensor:
        return F.create_affine_transformation_matrix(
            image_size, translation=self.translation, inverse_translation=self.inverse
        )

    def extra_grid_sample_repr(self) -> str:
        extras = ["factors={factors}"]
        if self.inverse:
            extras.append(["inverse={inverse}"])
        return ", ".join(extras).format(**self.__dict__)


class RGBToGrayscale(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rgb_to_grayscale(x)


class GrayscaleToFakegrayscale(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.grayscale_to_fakegrayscale(x)


class RGBToFakegrayscale(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rgb_to_fakegrayscale(x)


class GrayscaleToBinary(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.grayscale_to_binary(x)


class RGBToBinary(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rgb_to_binary(x)


class RGBToYUV(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rgb_to_yuv(x)


class YUVToRGB(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.yuv_to_rgb(x)
