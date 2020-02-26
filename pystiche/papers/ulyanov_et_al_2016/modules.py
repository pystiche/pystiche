from typing import Union, Tuple, Collection, Optional
import torch
from torch import nn
from torchvision.models.utils import load_state_dict_from_url

MODEL_URLS = {}

# FIXME: Check this
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


def get_norm_module(out_channels: int, instance_norm: bool) -> nn.Module:
    if instance_norm:
        return nn.InstanceNorm2d(out_channels)
    else:
        return nn.BatchNorm2d(out_channels)


def get_Noise(size: Tuple[int, int, int, int], channel: int = 3):
    return torch.rand(size[0], channel, size[2], size[3])


class UlyanovEtAl2016NoiseBlock(nn.Module):
    def __init__(
        self, noise_channel: int = 3, impl_params: bool = True, mode: str = "texture",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.impl_params = impl_params
        self.noise_channel = noise_channel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.mode == "style":
            if self.impl_params:
                return torch.cat((input, get_Noise(input.size(), channel=3)), 1)
        if self.mode == "texture":
            return get_Noise(input.size(), channel=3)
        return input


class UlyanovEtAl2016ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        impl_params: bool = True,
        stride: Union[Tuple[int, int], int] = 1,
        instance_norm: bool = True,
        inplace: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels

        def elementwise(fn, inputs):
            if isinstance(inputs, Collection):
                return tuple([fn(input) for input in inputs])
            return fn(inputs)

        def same_size_padding(kernel_size):
            return elementwise(lambda x: (x - 1) // 2, kernel_size)

        padding = same_size_padding(kernel_size)

        modules = []

        if padding > 0:
            modules.append(nn.ReflectionPad2d(padding))

        modules.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        )

        modules.append(get_norm_module(out_channels, instance_norm))

        if impl_params:
            activation = nn.ReLU(inplace=inplace)
        else:
            activation = nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
        modules.append(activation)

        super().__init__(*modules)


# def ulyanov_et_al_2016_conv_block(
#     in_channels: int,
#     out_channels: int,
#     impl_params: bool = True,
#     kernel_size: Union[Tuple[int, int], int] = 3,
#     stride: Union[Tuple[int, int], int] = 1,
#     instance_norm: bool = True,
#     inplace: bool = True,
# ) -> UlyanovEtAl2016ConvBlock:
#     return UlyanovEtAl2016ConvBlock(
#         in_channels,
#         out_channels,
#         impl_params=impl_params,
#         kernel_size=kernel_size,
#         stride=stride,
#         instance_norm=instance_norm,
#         inplace=inplace,
#     )


class UlyanovEtAl2016ConvSequence(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        impl_params: bool = True,
        instance_norm: bool = True,
        inplace: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        modules = (
            UlyanovEtAl2016ConvBlock(
                in_channels,
                out_channels,
                3,
                impl_params=impl_params,
                instance_norm=instance_norm,
                inplace=inplace,
            ),
            UlyanovEtAl2016ConvBlock(
                out_channels,
                out_channels,
                3,
                impl_params=impl_params,
                instance_norm=instance_norm,
                inplace=inplace,
            ),
            UlyanovEtAl2016ConvBlock(
                out_channels,
                out_channels,
                1,
                impl_params=impl_params,
                instance_norm=instance_norm,
                inplace=inplace,
            ),
        )

        super().__init__(*modules)


def ulyanov_et_al_2016_conv_sequence(
    in_channels: int,
    out_channels: int,
    impl_params: bool = True,
    instance_norm: bool = True,
    inplace: bool = True,
) -> UlyanovEtAl2016ConvSequence:
    return UlyanovEtAl2016ConvSequence(
        in_channels,
        out_channels,
        impl_params=impl_params,
        instance_norm=instance_norm,
        inplace=inplace,
    )


class UlaynovEtAl2016JoinBlock(nn.Module):
    def __init__(
        self,
        deep_block: UlyanovEtAl2016ConvSequence,
        shallow_block: UlyanovEtAl2016ConvSequence,
        instance_norm: bool = True,
    ) -> None:
        super().__init__()
        self.deep_block = deep_block
        self.deep_norm = get_norm_module(deep_block.out_channels, instance_norm)
        self.deep_upsample = nn.Upsample(scale_factor=2.0, mode="nearest")

        self.shallow_block = shallow_block
        self.shallow_norm = get_norm_module(shallow_block.out_channels, instance_norm)

        self.out_channels = self.deep_block.out_channels + shallow_block.out_channels

    def forward(
        self, deep_input: torch.Tensor, shallow_input: torch.Tensor
    ) -> torch.Tensor:
        deep_output = self.deep_norm(self.deep_upsample(self.deep_block(deep_input)))
        shallow_output = self.shallow_norm(self.shallow_block(shallow_input))
        return torch.cat((deep_output, shallow_output), dim=1)


class UlaynovEtAl2016LevelBlock(nn.Module):
    def __init__(
        self,
        deep_block: Union[UlyanovEtAl2016ConvSequence, "UlaynovEtAl2016LevelBlock"],
        shallow_block: UlyanovEtAl2016ConvSequence,
        impl_params: bool = True,
        instance_norm: bool = True,
        inplace: bool = True,
        mode: str = "texture",
    ):
        super().__init__()
        self.impl_params = impl_params
        self.mode = mode
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.noise_block = UlyanovEtAl2016NoiseBlock(
            noise_channel=3, impl_params=impl_params, mode=mode
        )
        if deep_block is not None:
            self.join = UlaynovEtAl2016JoinBlock(
                deep_block, shallow_block, instance_norm=instance_norm
            )
            out_channels = self.join.out_channels
            self.conv_sequence = UlyanovEtAl2016ConvSequence(
                out_channels,
                out_channels,
                impl_params=impl_params,
                instance_norm=instance_norm,
                inplace=inplace,
            )
        else:
            out_channels = 8
            self.join = None
            self.conv_sequence = shallow_block

        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.join is not None:
            deep_input = self.downsample(input)
            shallow_input = self.noise_block(input)
            join_output = self.join(deep_input, shallow_input)
        else:
            join_output = self.noise_block(input)
        return self.conv_sequence(join_output)


def ulyanov_et_al_2016_transformer(
    style: Optional[str] = None,
    weights: str = "pystiche",
    impl_params: bool = True,
    levels=5,
    instance_norm: bool = True,
    mode: str = "texture",
):
    levels = 6 if impl_params else levels
    input_channel = 6 if mode == "style" and not impl_params else 3
    level_block = None
    for _ in range(levels):
        level_block = UlaynovEtAl2016LevelBlock(
            level_block,
            ulyanov_et_al_2016_conv_sequence(
                input_channel, 8, impl_params=impl_params, instance_norm=instance_norm
            ),
            impl_params=impl_params,
            instance_norm=instance_norm,
            mode=mode,
        )

    modules = (
        level_block,
        nn.Conv2d(level_block.out_channels, 3, 1, stride=1),
    )
    transformer = nn.Sequential(*modules)
    if style is None:
        return transformer

    url = select_url(
        style, weights=weights, impl_params=impl_params, instance_norm=instance_norm
    )
    state_dict = load_state_dict_from_url(url)
    transformer.load_state_dict(state_dict)
    return transformer
