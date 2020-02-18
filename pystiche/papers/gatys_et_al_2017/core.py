from typing import Optional, Tuple, Dict
import logging
import torch
from torch import nn
from pystiche.pyramid import ImagePyramid
from pystiche.optim import default_image_pyramid_optim_loop
from ..common_utils import get_input_image
from .loss import (
    gatys_et_al_2017_perceptual_loss,
    gatys_et_al_2017_guided_perceptual_loss,
)
from .pyramid import gatys_et_al_2017_image_pyramid
from .utils import (
    gatys_et_al_2017_preprocessor,
    gatys_et_al_2017_postprocessor,
    gatys_et_al_2017_optimizer,
)

__all__ = [
    "gatys_et_al_2017_perceptual_loss",
    "gatys_et_al_2017_guided_perceptual_loss",
    "gatys_et_al_2017_image_pyramid",
    "gatys_et_al_2017_nst",
    "gatys_et_al_2017_guided_nst",
]


def gatys_et_al_2017_nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    impl_params: bool = True,
    criterion: Optional[nn.Module] = None,
    pyramid: Optional[ImagePyramid] = None,
    quiet: bool = False,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    if criterion is None:
        criterion = gatys_et_al_2017_perceptual_loss(impl_params=impl_params)

    if pyramid is None:
        pyramid = gatys_et_al_2017_image_pyramid(resize_targets=(criterion,))

    device = content_image.device
    criterion = criterion.to(device)

    initial_resize = pyramid[-1].resize_image
    content_image = initial_resize(content_image)
    style_image = initial_resize(style_image)
    input_image = get_input_image(starting_point="content", content_image=content_image)

    preprocessor = gatys_et_al_2017_preprocessor().to(device)
    postprocessor = gatys_et_al_2017_postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))
    criterion.set_style_image(preprocessor(style_image))

    return default_image_pyramid_optim_loop(
        input_image,
        criterion,
        pyramid,
        get_optimizer=gatys_et_al_2017_optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
    )


def gatys_et_al_2017_guided_nst(
    content_image: torch.Tensor,
    content_guides: Dict[str, torch.Tensor],
    style_images_and_guides: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    impl_params: bool = True,
    criterion: Optional[nn.Module] = None,
    pyramid: Optional[ImagePyramid] = None,
    quiet: bool = False,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    regions = set(content_guides.keys())
    if regions != set(style_images_and_guides.keys()):
        # FIXME
        raise RuntimeError
    regions = sorted(regions)

    if criterion is None:
        criterion = gatys_et_al_2017_guided_perceptual_loss(
            regions, impl_params=impl_params
        )

    if pyramid is None:
        pyramid = gatys_et_al_2017_image_pyramid(resize_targets=(criterion,))

    device = content_image.device
    criterion = criterion.to(device)

    initial_image_resize = pyramid[-1].resize_image
    initial_guide_resize = pyramid[-1].resize_guide

    content_image = initial_image_resize(content_image)
    content_guides = {
        region: initial_guide_resize(guide) for region, guide in content_guides.items()
    }
    style_images_and_guides = {
        region: (initial_image_resize(image), initial_guide_resize(guide))
        for region, (image, guide) in style_images_and_guides.items()
    }
    input_image = get_input_image(starting_point="content", content_image=content_image)

    preprocessor = gatys_et_al_2017_preprocessor().to(device)
    postprocessor = gatys_et_al_2017_postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))

    for region, (image, guide) in style_images_and_guides.items():
        criterion.set_style_guide(region, guide)
        criterion.set_style_image(region, preprocessor(image))

    for region, guide in content_guides.items():
        criterion.set_content_guide(region, guide)

    return default_image_pyramid_optim_loop(
        input_image,
        criterion,
        pyramid,
        get_optimizer=gatys_et_al_2017_optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
    )
