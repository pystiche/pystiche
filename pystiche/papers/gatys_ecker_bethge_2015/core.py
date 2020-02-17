from typing import Optional, Callable
import logging
import torch
from torch import nn
from pystiche.optim import default_image_optim_loop
from .utils import (
    gatys_ecker_bethge_2015_preprocessor,
    gatys_ecker_bethge_2015_postprocessor,
    gatys_ecker_bethge_2015_optimizer,
)
from ..common_utils import get_input_image
from .loss import gatys_ecker_bethge_2015_perceptual_loss


__all__ = [
    "gatys_ecker_bethge_2015_perceptual_loss",
    "gatys_ecker_bethge_2015_nst",
]


def gatys_ecker_bethge_2015_nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    num_steps: int = 500,
    impl_params: bool = True,
    criterion: Optional[nn.Module] = None,
    quiet: bool = False,
    logger: Optional[logging.Logger] = None,
    log_fn: Optional[Callable] = None,
) -> torch.Tensor:
    if criterion is None:
        criterion = gatys_ecker_bethge_2015_perceptual_loss(impl_params=impl_params)

    device = content_image.device
    criterion = criterion.to(device)

    starting_point = "content" if impl_params else "random"
    input_image = get_input_image(
        starting_point=starting_point, content_image=content_image
    )

    preprocessor = gatys_ecker_bethge_2015_preprocessor().to(device)
    postprocessor = gatys_ecker_bethge_2015_postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))
    criterion.set_style_image(preprocessor(style_image))

    return default_image_optim_loop(
        input_image,
        criterion,
        get_optimizer=gatys_ecker_bethge_2015_optimizer,
        num_steps=num_steps,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
    )
