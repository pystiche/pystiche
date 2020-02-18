from typing import Optional
import logging
import torch
from torch import nn
from pystiche.pyramid import ImagePyramid
from pystiche.optim import default_image_pyramid_optim_loop
from ..common_utils import get_input_image
from .data import li_wand_2016_images
from .loss import li_wand_2016_perceptual_loss
from .pyramid import li_wand_2016_image_pyramid
from .utils import (
    li_wand_2016_preprocessor,
    li_wand_2016_postprocessor,
    li_wand_2016_optimizer,
)


__all__ = [
    "li_wand_2016_images",
    "li_wand_2016_perceptual_loss",
    "li_wand_2016_image_pyramid",
    "li_wand_2016_nst",
]


def li_wand_2016_nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    impl_params: bool = True,
    criterion: Optional[nn.Module] = None,
    pyramid: Optional[ImagePyramid] = None,
    quiet: bool = False,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    if criterion is None:
        criterion = li_wand_2016_perceptual_loss(impl_params=impl_params)

    if pyramid is None:
        pyramid = li_wand_2016_image_pyramid(resize_targets=(criterion,))

    device = content_image.device
    criterion = criterion.to(device)

    initial_resize = pyramid[-1].resize_image
    content_image = initial_resize(content_image)
    style_image = initial_resize(style_image)
    input_image = get_input_image(starting_point="content", content_image=content_image)

    preprocessor = li_wand_2016_preprocessor().to(device)
    postprocessor = li_wand_2016_postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))
    criterion.set_style_image(preprocessor(style_image))

    return default_image_pyramid_optim_loop(
        input_image,
        criterion,
        pyramid,
        get_optimizer=li_wand_2016_optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
    )
