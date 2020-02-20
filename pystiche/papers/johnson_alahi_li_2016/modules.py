from typing import Union, Optional, Tuple
from math import sqrt
import torch
from torch import nn
from torchvision.models.utils import load_state_dict_from_url
import pystiche
from ..common_utils import (
    same_size_padding,
    same_size_output_padding,
    ResidualBlock,
)

__all__ = [
    "johnson_alahi_li_2016_conv_block",
    "johnson_alahi_li_2016_residual_block",
    "johnson_alahi_li_2016_transformer_encoder",
    "johnson_alahi_li_2016_transformer_decoder",
    "JohnsonAlahiLi2016Transformer",
    "johnson_alahi_li_2016_transformer",
]

MODEL_URLS = {
    (
        "author",
        "candy__impl_params__instance_norm",
    ): "johnson_alahi_li_2016_transformer__candy__impl_params__instance_norm-48e84161.pth",
    (
        "author",
        "composition_vii__impl_params",
    ): "johnson_alahi_li_2016_transformer__composition_vii__impl_params-2cff0b7b.pth",
    (
        "author",
        "feathers__impl_params__instance_norm",
    ): "johnson_alahi_li_2016_transformer__feathers__impl_params__instance_norm-21932db0.pth",
    (
        "author",
        "la_muse__impl_params",
    ): "johnson_alahi_li_2016_transformer__la_muse__impl_params-3ef8efb0.pth",
    (
        "author",
        "la_muse__impl_params__instance_norm",
    ): "johnson_alahi_li_2016_transformer__la_muse__impl_params__instance_norm-f553d8b7.pth",
    (
        "author",
        "mosaic__impl_params__instance_norm",
    ): "johnson_alahi_li_2016_transformer__mosaic__impl_params__instance_norm-c614f29e.pth",
    (
        "author",
        "starry_night__impl_params",
    ): "johnson_alahi_li_2016_transformer__starry_night__impl_params-b6422c84.pth",
    (
        "author",
        "the_scream__impl_params__instance_norm",
    ): "johnson_alahi_li_2016_transformer__the_scream__impl_params__instance_norm-ee572f90.pth",
    (
        "author",
        "the_wave__impl_params",
    ): "johnson_alahi_li_2016_transformer__the_wave__impl_params-63ad42b2.pth",
    (
        "author",
        "udnie__impl_params__instance_norm",
    ): "johnson_alahi_li_2016_transformer__udnie__impl_params__instance_norm-aa5e1467.pth",
}


def select_url(style: str, weights: str, impl_params: bool, instance_norm: bool):
    opt = style
    if impl_params:
        opt += "__impl_params"
    if instance_norm:
        opt += "__instance_norm"
    for (valid_weights, valid_opt), url in MODEL_URLS.items():
        if weights == valid_weights and valid_opt.startswith(opt):
            return url
    else:
        raise RuntimeError


def get_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int] = 1,
    padding: Optional[Union[Tuple[int, int], int]] = None,
    upsample: bool = False,
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    if padding is None:
        padding = same_size_padding(kernel_size)
    if not upsample:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        )
    else:
        output_padding = same_size_output_padding(stride)
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )


def get_norm(
    out_channels: int, instance_norm: bool
) -> Union[nn.BatchNorm2d, nn.InstanceNorm2d]:
    norm_kwargs = {
        "eps": 1e-5,
        "momentum": 1e-1,
        "affine": True,
        "track_running_stats": True,
    }
    if instance_norm:
        return nn.InstanceNorm2d(out_channels, **norm_kwargs)
    else:
        return nn.BatchNorm2d(out_channels, **norm_kwargs)


def johnson_alahi_li_2016_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int] = 1,
    padding: Optional[Union[Tuple[int, int], int]] = None,
    upsample: bool = False,
    relu: bool = True,
    inplace: bool = True,
    instance_norm: bool = True,
) -> nn.Sequential:
    modules = [
        get_conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            upsample=upsample,
        ),
        get_norm(out_channels, instance_norm),
    ]
    if relu:
        modules.append(nn.ReLU(inplace=inplace))
    return nn.Sequential(*modules)


def johnson_alahi_li_2016_residual_block(
    channels: int, inplace: bool = True, instance_norm: bool = True,
) -> ResidualBlock:
    in_channels = out_channels = channels
    kernel_size = 3
    residual = nn.Sequential(
        johnson_alahi_li_2016_conv_block(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            inplace=inplace,
            instance_norm=instance_norm,
        ),
        johnson_alahi_li_2016_conv_block(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            instance_norm=instance_norm,
            relu=False,
        ),
    )

    class CenterCrop(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, :, 2:-2, 2:-2]

    shortcut = CenterCrop()

    return ResidualBlock(residual, shortcut)


def johnson_alahi_li_2016_transformer_encoder(
    instance_norm: bool = True,
) -> pystiche.SequentialModule:
    modules = (
        nn.ReflectionPad2d(40),
        johnson_alahi_li_2016_conv_block(
            in_channels=3,
            out_channels=16 if instance_norm else 32,
            kernel_size=9,
            instance_norm=instance_norm,
        ),
        johnson_alahi_li_2016_conv_block(
            in_channels=16 if instance_norm else 32,
            out_channels=32 if instance_norm else 64,
            kernel_size=3,
            stride=2,
            instance_norm=instance_norm,
        ),
        johnson_alahi_li_2016_conv_block(
            in_channels=32 if instance_norm else 64,
            out_channels=64 if instance_norm else 128,
            kernel_size=3,
            stride=2,
            instance_norm=instance_norm,
        ),
        johnson_alahi_li_2016_residual_block(
            channels=64 if instance_norm else 128, instance_norm=instance_norm,
        ),
        johnson_alahi_li_2016_residual_block(
            channels=64 if instance_norm else 128, instance_norm=instance_norm,
        ),
        johnson_alahi_li_2016_residual_block(
            channels=64 if instance_norm else 128, instance_norm=instance_norm,
        ),
        johnson_alahi_li_2016_residual_block(
            channels=64 if instance_norm else 128, instance_norm=instance_norm,
        ),
        johnson_alahi_li_2016_residual_block(
            channels=64 if instance_norm else 128, instance_norm=instance_norm,
        ),
    )
    return pystiche.SequentialModule(*modules)


def johnson_alahi_li_2016_transformer_decoder(
    impl_params: bool = True, instance_norm: bool = True,
) -> pystiche.SequentialModule:
    def get_value_range_delimiter() -> nn.Module:
        if impl_params:

            def value_range_delimiter(x: torch.Tensor) -> torch.Tensor:
                return 150.0 * torch.tanh(x)

        else:

            def value_range_delimiter(x: torch.Tensor) -> torch.Tensor:
                # sgm(2*x) == (tanh(x) + 1) / 2
                return torch.sigmoid(2.0 * x)

        class ValueRangeDelimiter(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return value_range_delimiter(x)

        return ValueRangeDelimiter()

    modules = (
        johnson_alahi_li_2016_conv_block(
            in_channels=64 if instance_norm else 128,
            out_channels=32 if instance_norm else 64,
            kernel_size=3,
            stride=2,
            upsample=True,
            instance_norm=instance_norm,
        ),
        johnson_alahi_li_2016_conv_block(
            in_channels=32 if instance_norm else 64,
            out_channels=16 if instance_norm else 32,
            kernel_size=3,
            stride=2,
            upsample=True,
            instance_norm=instance_norm,
        ),
        nn.Conv2d(
            in_channels=16 if instance_norm else 32,
            out_channels=3,
            kernel_size=9,
            padding=same_size_padding(kernel_size=9),
        ),
        get_value_range_delimiter(),
    )

    return pystiche.SequentialModule(*modules)


class JohnsonAlahiLi2016Transformer(nn.Module):
    def __init__(
        self,
        impl_params: bool = True,
        instance_norm: bool = True,
        init_weights: bool = True,
    ):
        super().__init__()
        self.encoder = johnson_alahi_li_2016_transformer_encoder(
            instance_norm=instance_norm
        )
        self.decoder = johnson_alahi_li_2016_transformer_decoder(
            impl_params=impl_params, instance_norm=instance_norm
        )
        if init_weights:
            self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / sqrt(fan_in)
                nn.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -bound, bound)
            if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def johnson_alahi_li_2016_transformer(
    style: Optional[str] = None,
    weights: str = "pystiche",
    impl_params: bool = True,
    instance_norm: Optional[bool] = None,
) -> JohnsonAlahiLi2016Transformer:
    if instance_norm is None:
        instance_norm = impl_params

    if instance_norm and not impl_params:
        raise RuntimeError

    init_weights = style is None
    transformer = JohnsonAlahiLi2016Transformer(
        impl_params=impl_params, instance_norm=instance_norm, init_weights=init_weights
    )
    if init_weights:
        return transformer

    url = select_url(
        style, weights=weights, impl_params=impl_params, instance_norm=instance_norm
    )
    state_dict = load_state_dict_from_url(url)
    transformer.load_state_dict(state_dict)
    return transformer
