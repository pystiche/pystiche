from typing import Union, Optional, Tuple
from PIL import Image
import torch
from torchvision.transforms.functional import (
    to_tensor as _to_tensor,
    to_pil_image as _to_pil_image,
)
from pystiche.image.utils import (
    is_batched_image,
    extract_batch_size,
    make_batched_image,
    make_single_image,
    force_image,
)

__all__ = [
    "import_from_pil",
    "export_to_pil",
]


def import_from_pil(
    image: Image.Image,
    device: Union[torch.device, str] = "cpu",
    make_batched: bool = True,
) -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)
    image = _to_tensor(image).to(device)
    if make_batched:
        image = make_batched_image(image)
    return image


@force_image
def export_to_pil(
    image: torch.Tensor, mode: Optional[str] = None
) -> Union[Image.Image, Tuple[Image.Image, ...]]:
    def fn(image: torch.Tensor) -> Image.Image:
        return _to_pil_image(image.detach().cpu().clamp(0.0, 1.0), mode)

    if is_batched_image(image):
        batched_image = image
        batch_size = extract_batch_size(batched_image)
        if batch_size == 1:
            return fn(make_single_image(batched_image))
        else:
            return tuple([fn(single_image) for single_image in batched_image])

    return fn(image)
