from typing import Union, Optional, Callable
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
from pystiche.optim import OptimLogger, default_transformer_optim_loop
from ..common_utils import batch_up_image
from .modules import JohnsonAlahiLi2016Transformer, johnson_alahi_li_2016_transformer
from .loss import (
    JohnsonAlahiLi2016PerceptualLoss,
    johnson_alahi_li_2016_perceptual_loss,
)
from .data import (
    johnson_alahi_li_2016_style_transform,
    johnson_alahi_li_2016_dataset,
    johnson_alahi_li_2016_image_loader,
    johnson_alahi_li_2016_images,
)
from .utils import (
    johnson_alahi_li_2016_preprocessor,
    johnson_alahi_li_2016_postprocessor,
    johnson_alahi_li_2016_optimizer,
)

__all__ = [
    "johnson_alahi_li_2016_transformer",
    "johnson_alahi_li_2016_perceptual_loss",
    "johnson_alahi_li_2016_dataset",
    "johnson_alahi_li_2016_image_loader",
    "johnson_alahi_li_2016_training",
    "johnson_alahi_li_2016_images",
    "johnson_alahi_li_2016_stylization",
]


def johnson_alahi_li_2016_training(
    content_image_loader: DataLoader,
    style: Union[str, torch.Tensor],
    impl_params=True,
    instance_norm: bool = True,
    transformer: Optional[JohnsonAlahiLi2016Transformer] = None,
    criterion: Optional[JohnsonAlahiLi2016PerceptualLoss] = None,
    get_optimizer: Optional[
        Callable[[JohnsonAlahiLi2016Transformer], Optimizer]
    ] = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
):
    if isinstance(style, str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = johnson_alahi_li_2016_images(download=False)
        style_image = images[style].read(device=device)
    else:
        style_image = style
        device = style_image.device
        style = None

    if impl_params:
        preprocessor = johnson_alahi_li_2016_preprocessor()
        preprocessor = preprocessor.to(device)
        style_image = preprocessor(style_image)

    if transformer is None:
        transformer = johnson_alahi_li_2016_transformer(
            impl_params=impl_params, instance_norm=instance_norm
        )
        transformer = transformer.train()
    transformer = transformer.to(device)

    if criterion is None:
        criterion = johnson_alahi_li_2016_perceptual_loss(
            impl_params=impl_params, instance_norm=instance_norm, style=style
        )
        criterion = criterion.eval()
    criterion = criterion.to(device)

    if get_optimizer is None:
        get_optimizer = johnson_alahi_li_2016_optimizer

    style_transform = johnson_alahi_li_2016_style_transform(
        impl_params=impl_params, instance_norm=instance_norm, style=style
    )
    style_transform = style_transform.to(device)
    style_image = style_transform(style_image)
    style_image = batch_up_image(style_image, loader=content_image_loader)

    criterion.set_style_image(style_image)

    def criterion_update_fn(input_image, criterion):
        criterion.set_content_image(input_image)

    default_transformer_optim_loop(
        content_image_loader,
        device,
        transformer,
        criterion,
        criterion_update_fn,
        get_optimizer=get_optimizer,
        quiet=quiet,
        logger=logger,
    )

    return transformer


def johnson_alahi_li_2016_stylization(
    input_image: torch.Tensor,
    transformer: Union[nn.Module, str],
    impl_params: bool = True,
    instance_norm: bool = None,
    weights: str = "pystiche",
    preprocessor: Optional[nn.Module] = None,
    postprocessor: Optional[nn.Module] = None,
):
    device = input_image.device

    if isinstance(transformer, str):
        style = transformer
        transformer = johnson_alahi_li_2016_transformer(
            style=style,
            weights=weights,
            impl_params=impl_params,
            instance_norm=instance_norm,
        )
        transformer = transformer.eval()
        transformer = transformer.to(device)

    with torch.no_grad():
        if preprocessor is None:
            preprocessor = johnson_alahi_li_2016_preprocessor()
            preprocessor = preprocessor.to(device)

            input_image = preprocessor(input_image)

        output_image = transformer(input_image)

        if postprocessor is None:
            postprocessor = johnson_alahi_li_2016_postprocessor()
            postprocessor = postprocessor.to(device)

            output_image = postprocessor(output_image)

    return output_image.detach()
