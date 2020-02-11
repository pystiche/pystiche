from typing import Union, Optional, Sequence, Tuple
from PIL import Image
import torch
from torch.nn.functional import interpolate
from torchvision.transforms.functional import (
    to_tensor as _to_tensor,
    to_pil_image as _to_pil_image,
)
from pystiche.image.utils import (
    verify_is_single_image,
    is_single_image,
    verify_is_image,
    apply_imagewise,
    extract_batch_size,
)
from pystiche.typing import Numeric
from ._utils import get_align_corners

__all__ = [
    "add_batch_dim",
    "remove_batch_dim",
    "import_from_pil",
    "export_to_pil",
    "normalize",
    "denormalize",
    "float_to_uint8_range",
    "uint8_to_float_range",
    "reverse_channel_order",
    "resize",
    "transform_channels_linearly",
    "transform_channels_affinely",
]


def add_batch_dim(x: torch.Tensor) -> torch.Tensor:
    verify_is_single_image(x)
    return x.unsqueeze(0)


def remove_batch_dim(x: torch.Tensor) -> torch.Tensor:
    batch_size = extract_batch_size(x)
    if batch_size != 1:
        msg = "ADDME"  # FIXME
        raise RuntimeError(msg)
    return x.squeeze(0)


def import_from_pil(
    image: Image.Image,
    device: Union[torch.device, str] = "cpu",
    make_batched: bool = True,
) -> torch.FloatTensor:
    if isinstance(device, str):
        device = torch.device(device)
    image = _to_tensor(image).to(device)
    if make_batched:
        image = add_batch_dim(image)
    return image


def export_to_pil(
    image: torch.Tensor, mode: Optional[str] = None
) -> Union[Image.Image, Tuple[Image.Image, ...]]:
    def fn(image: torch.Tensor) -> Image.Image:
        return _to_pil_image(image.detach().cpu().clamp(0.0, 1.0), mode)

    return apply_imagewise(fn, image)


def float_to_uint8_range(x: torch.Tensor) -> torch.Tensor:
    return x * 255.0


def uint8_to_float_range(x: torch.Tensor) -> torch.Tensor:
    return x / 255.0


def reverse_channel_order(x: torch.Tensor) -> torch.Tensor:
    return x.flip((1,))


def denormalize(x: torch.Tensor, mean: Numeric, std: Numeric) -> torch.Tensor:
    return x * std + mean


def normalize(x: torch.Tensor, mean: Numeric, std: Numeric) -> torch.Tensor:
    return (x - mean) / std


def resize(
    x: torch.Tensor, size: Sequence[int], interpolation_mode: str = "bilinear"
) -> torch.Tensor:
    return interpolate(
        x,
        size=size,
        scale_factor=None,
        mode=interpolation_mode,
        align_corners=get_align_corners(interpolation_mode),
    )


def transform_channels_linearly(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return transform_channels_affinely(x, matrix, bias=None)


def transform_channels_affinely(
    x: torch.Tensor, matrix: torch.Tensor, bias: Optional[torch.tensor] = None
) -> torch.Tensor:
    batch_size, _, *spatial_size = x.size()
    x = torch.flatten(x, 2)

    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    num_channels = matrix.size()[1]

    x = torch.bmm(matrix, x)

    if bias is not None:
        if bias.dim() == 2:
            bias = bias.unsqueeze(0)
        x += bias

    return x.view(batch_size, num_channels, *spatial_size)
