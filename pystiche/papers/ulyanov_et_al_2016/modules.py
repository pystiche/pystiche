from typing import Any, Union, Tuple, Collection, Sequence, Optional
from typing_extensions import Protocol
from collections import OrderedDict
from math import sqrt
import torch
from torch import nn
from torchvision.models.utils import load_state_dict_from_url
import pystiche
from pystiche.misc import zip_equal

MODEL_URLS = {}


def select_url(style: str, impl_params: bool, instance_norm: bool) -> str:
    # FIXME: implement this
    raise RuntimeError
    # opt = style
    # if impl_params:
    #     opt += "__impl_params"
    # if instance_norm:
    #     opt += "__instance_norm"
    # for (valid_weights, valid_opt), url in MODEL_URLS.items():
    #     if weights == valid_weights and valid_opt.startswith(opt):
    #         return url
    # else:
    #     raise RuntimeError


def get_norm_module(
    in_channels: int, instance_norm: bool
) -> Union[nn.BatchNorm2d, nn.InstanceNorm2d]:
    norm_kwargs = {
        "eps": 1e-5,
        "momentum": 1e-1,
        "affine": True,
        "track_running_stats": True,
    }
    if instance_norm:
        return nn.InstanceNorm2d(in_channels, **norm_kwargs)
    else:
        return nn.BatchNorm2d(in_channels, **norm_kwargs)


def get_activation_module(impl_params: bool, inplace: bool = True):
    if impl_params:
        return nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
    else:
        return nn.LeakyReLU(negative_slope=0.01, inplace=inplace)


def join_channelwise(*inputs: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
    return torch.cat(inputs, dim=channel_dim)


class UlyanovEtAl2016JoinBlock(nn.Module):
    def __init__(
        self,
        branch_in_channels: Sequence[int],
        instance_norm: bool = True,
        channel_dim: int = 1,
    ) -> None:
        super().__init__()

        # FIXME: make name optional
        name_fmt = "join" + r"{:0" + str(len(str(len(branch_in_channels)))) + r"}"
        norm_modules = []
        for idx, in_channels in enumerate(branch_in_channels):
            norm_module = get_norm_module(in_channels, instance_norm)
            self.add_module(name_fmt.format(idx), norm_module)
            norm_modules.append(norm_module)
        self.norm_modules = norm_modules

        self.channel_dim = channel_dim

    @property
    def out_channels(self) -> int:
        return sum([norm.num_features for norm in self.norm_modules])

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return join_channelwise(
            *[norm(input) for norm, input in zip_equal(self.norm_modules, inputs)],
            channel_dim=self.channel_dim
        )


class SequentialWithOutChannels(nn.Sequential):
    def __init__(self, *args, out_channel_name: Optional[Union[str, int]] = None):
        super().__init__(*args)
        if out_channel_name is None:
            out_channel_name = tuple(self._modules.keys())[-1]
        elif isinstance(out_channel_name, int):
            out_channel_name = str(out_channel_name)

        self.out_channels = self._modules[out_channel_name].out_channels


class NoiseFn(Protocol):
    def __call__(self, size: torch.Size, **meta: Any) -> torch.Tensor:
        pass


class UlyanovEtAl2016NoiseModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_noise_channels: int = 3,
        noise_fn: Optional[NoiseFn] = None,
    ):
        super().__init__()
        self.num_noise_channels = num_noise_channels

        if noise_fn is None:
            noise_fn = torch.rand
        self.noise_fn = noise_fn

        self.in_channels = in_channels
        self.out_channels = in_channels + num_noise_channels


class UlyanovEtAl2016StylizationNoise(UlyanovEtAl2016NoiseModule):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        def get_size(input: torch.Tensor) -> torch.Size:
            size = list(input.size())
            size[1] = self.num_noise_channels
            return torch.Size(size)

        size = get_size(input)
        meta = pystiche.tensor_meta(input)
        noise = self.noise_fn(size, **meta)
        return join_channelwise(input, noise)


# class UlyanovEtAl2016TextureNoiseGeneration(UlyanovEtAl2016NoiseModule):
#     def forward(
#         self,
#         input: Union[Tuple[int, int], torch.Tensor],
#         batch_size: int = 1,
#         **meta: Any
#     ) -> torch.Tensor:
#         def get_size(input: Union[Tuple[int, int], torch.Tensor]) -> torch.Size:
#             # FIXME: downsample
#             if isinstance(input, torch.Tensor):
#                 image_size = extract_image_size(input)
#             else:
#                 image_size = input
#             return torch.Size((batch_size, self.num_noise_channels, *image_size))
#
#         def get_meta(input: Union[Tuple[int, int], torch.Tensor]) -> Dict[str, Any]:
#             if isinstance(input, torch.Tensor):
#                 return pystiche.tensor_meta(input, **meta)
#             else:
#                 return meta
#
#         size = get_size(input)
#         meta = get_meta(input)
#         return self.noise_fn(size, **meta)


def ulyanov_et_al_2016_noise(
    stylization: bool = True,
    in_channels: int = 3,
    num_noise_channels: int = 3,
    noise_fn: Optional[NoiseFn] = None,
) -> UlyanovEtAl2016NoiseModule:
    if stylization:
        return UlyanovEtAl2016StylizationNoise(
            in_channels, num_noise_channels=num_noise_channels, noise_fn=noise_fn
        )
    else:
        raise RuntimeError


class UlyanovEtAl2016StylizationDownsample(nn.AvgPool2d):
    def __init__(self, in_channels: int, kernel_size=2, stride=2, padding=0):
        super().__init__(kernel_size, stride=stride, padding=padding)

        self.in_channels = self.out_channels = in_channels


def ulyanov_et_al_2016_downsample(
    stylization: bool = True, in_channels: int = 3
) -> nn.Module:
    if stylization:
        return UlyanovEtAl2016StylizationDownsample(in_channels)
    else:
        raise RuntimeError


def ulyanov_et_al_2016_upsample() -> nn.Upsample:
    return nn.Upsample(scale_factor=2.0, mode="nearest")


class UlyanovEtAl2016HourGlassBlock(SequentialWithOutChannels):
    def __init__(
        self, in_channels: int, intermediate: nn.Module, stylization: bool = True,
    ):
        modules = (
            (
                "down",
                ulyanov_et_al_2016_downsample(
                    stylization=stylization, in_channels=in_channels
                ),
            ),
            ("intermediate", intermediate),
            ("up", ulyanov_et_al_2016_upsample()),
        )
        super().__init__(OrderedDict(modules), out_channel_name="intermediate")


class UlyanovEtAl2016ConvBlock(SequentialWithOutChannels):
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
        # FIXME: use from commmon utils
        def elementwise(fn, inputs):
            if isinstance(inputs, Collection):
                return tuple([fn(input) for input in inputs])
            return fn(inputs)

        def same_size_padding(kernel_size):
            return elementwise(lambda x: (x - 1) // 2, kernel_size)

        padding = same_size_padding(kernel_size)

        modules = []

        # FIXME: is_valid_padding
        if padding:
            modules.append(("pad", nn.ReflectionPad2d(padding)))

        modules.append(
            (
                "conv",
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride=stride, padding=0
                ),
            )
        )
        modules.append(("norm", get_norm_module(out_channels, instance_norm)))
        modules.append(("act", get_activation_module(impl_params, inplace=inplace)))

        super().__init__(OrderedDict(modules), out_channel_name="conv")


class UlyanovEtAl2016ConvSequence(SequentialWithOutChannels):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        impl_params: bool = True,
        instance_norm: bool = True,
        inplace: bool = True,
    ):
        def conv_block(
            in_channels: int, out_channels: int, kernel_size: int
        ) -> UlyanovEtAl2016ConvBlock:
            return UlyanovEtAl2016ConvBlock(
                in_channels,
                out_channels,
                kernel_size,
                impl_params=impl_params,
                instance_norm=instance_norm,
                inplace=inplace,
            )

        modules = [
            ("conv_block1", conv_block(in_channels, out_channels, kernel_size=3)),
            ("conv_block2", conv_block(out_channels, out_channels, kernel_size=3)),
        ]

        if not impl_params:
            modules.append(
                ("conv_block3", conv_block(out_channels, out_channels, kernel_size=1))
            )

        super().__init__(OrderedDict(modules))


class UlyanovEtAl2016BranchBlock(nn.Module):
    def __init__(
        self,
        deep_branch: nn.Module,
        shallow_branch: nn.Module,
        instance_norm: bool = True,
    ):
        super().__init__()
        self.deep = deep_branch
        self.shallow = shallow_branch
        self.join = UlyanovEtAl2016JoinBlock(
            (deep_branch.out_channels, shallow_branch.out_channels),
            instance_norm=instance_norm,
        )

    @property
    def out_channels(self) -> int:
        return self.join.out_channels

    def forward(self, input: Any, **kwargs: Any) -> torch.Tensor:
        deep_output = self.deep(input, **kwargs)
        shallow_output = self.shallow(input, **kwargs)
        return self.join(deep_output, shallow_output)


def ulyanov_et_al_2016_level(
    prev_level_block: Optional[SequentialWithOutChannels],
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    in_channels: int = 3,
    num_noise_channels: int = 3,
    noise_fn: Optional[NoiseFn] = None,
    inplace: bool = True,
) -> SequentialWithOutChannels:
    def conv_sequence(in_channels: int, out_channels: int, noise: bool = False):
        modules = []

        if noise:
            noise_module = ulyanov_et_al_2016_noise(
                stylization,
                in_channels=in_channels,
                num_noise_channels=num_noise_channels,
                noise_fn=noise_fn,
            )
            in_channels = noise_module.out_channels
            modules.append(("noise", noise_module))

        conv_seq = UlyanovEtAl2016ConvSequence(
            in_channels,
            out_channels,
            impl_params=impl_params,
            instance_norm=instance_norm,
            inplace=inplace,
        )

        if not noise:
            return conv_seq

        modules.append(("conv_seq", conv_seq))
        return SequentialWithOutChannels(OrderedDict(modules))

    shallow_brach = conv_sequence(in_channels, out_channels=8, noise=True)

    if prev_level_block is None:
        return shallow_brach

    deep_branch = UlyanovEtAl2016HourGlassBlock(
        in_channels, prev_level_block, stylization=stylization,
    )
    branch_block = UlyanovEtAl2016BranchBlock(
        deep_branch, shallow_brach, instance_norm=instance_norm
    )

    output_conv_seq = conv_sequence(
        branch_block.out_channels, branch_block.out_channels
    )

    return SequentialWithOutChannels(
        OrderedDict((("branch", branch_block), ("output_conv_seq", output_conv_seq)))
    )


class UlyanovEtAl2016Transformer(nn.Sequential):
    def __init__(
        self,
        levels: int,
        impl_params: bool = True,
        instance_norm: bool = True,
        stylization: bool = True,
        init_weights: bool = True,
    ) -> None:
        pyramid = None
        for _ in range(levels):
            pyramid = ulyanov_et_al_2016_level(
                pyramid,
                impl_params=impl_params,
                instance_norm=instance_norm,
                stylization=stylization,
            )

        output_conv = nn.Conv2d(
            pyramid.out_channels, out_channels=3, kernel_size=1, stride=1, padding=0
        )

        super().__init__(
            OrderedDict((("pyramid", pyramid), ("output_conv", output_conv)))
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


def ulyanov_et_al_2016_transformer(
    style: Optional[str] = None,
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    levels: Optional[int] = None,
):
    if levels is None:
        if stylization:
            levels = 5 if impl_params else 6

    init_weights = style is None
    transformer = UlyanovEtAl2016Transformer(
        levels,
        impl_params=impl_params,
        instance_norm=instance_norm,
        stylization=stylization,
        init_weights=init_weights,
    )
    if init_weights:
        return transformer

    url = select_url(style, impl_params=impl_params, instance_norm=instance_norm)
    state_dict = load_state_dict_from_url(url)
    transformer.load_state_dict(state_dict)
    return transformer
