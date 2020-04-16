from torch import nn
from typing import Union, Tuple, Collection, List
import pystiche
import torch
from ..common_utils import ResidualBlock


def get_norm_module(out_channels: int, instance_norm: bool) -> nn.Module:
    if instance_norm:
        return nn.InstanceNorm2d(out_channels)
    else:
        return nn.BatchNorm2d(out_channels)


def get_padding(padding: str, kernel_size: Union[Tuple[int, int], int]) -> int:
    def elementwise(fn, inputs):
        if isinstance(inputs, Collection):
            return tuple([fn(input) for input in inputs])
        return fn(inputs)

    def same_size_padding(kernel_size):
        return elementwise(lambda x: (x - 1) // 2, kernel_size)

    if padding == "same":
        return same_size_padding(kernel_size)
    elif padding == "valid":
        return 0
    else:
        raise ValueError


def sanakoyeu_et_al_2018_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int] = 2,
    padding: str = "valid",
) -> nn.Conv2d:
    padding = get_padding(padding, kernel_size)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding
    )


class SanakoyeuEtAl2018ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int] = 1,
        padding: str = "valid",
        instance_norm: bool = True,
        use_act: bool = True,
        act: str = "relu",
        inplace: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels

        modules = []
        modules.append(
            sanakoyeu_et_al_2018_conv(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        )

        modules.append(get_norm_module(out_channels, instance_norm))

        if use_act:
            if act == "relu":
                activation = nn.ReLU(inplace=inplace)
            elif act == "lrelu":
                activation = nn.LeakyReLU(negative_slope=0.2, inplace=inplace)
            else:
                raise NotImplementedError

            modules.append(activation)

        super().__init__(*modules)


class SanakoyeuEtAl2018ConvTransponse(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int] = 2,
        padding: str = "valid",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        padding = get_padding(padding, kernel_size)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding
        )  # TODO: init weights and bias None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(
            nn.functional.interpolate(input, scale_factor=self.stride, mode="nearest")
        )


class SanakoyeuEtAl2018ConvTransponseBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int] = 1,
        padding: str = "valid",
        instance_norm: bool = True,
        use_act: bool = True,
        act: str = "relu",
        inplace: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels

        modules = []
        modules.append(
            SanakoyeuEtAl2018ConvTransponse(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        )

        modules.append(get_norm_module(out_channels, instance_norm))

        if use_act:
            if act == "relu":
                activation = nn.ReLU(inplace=inplace)
            elif act == "lrelu":
                activation = nn.LeakyReLU(negative_slope=0.2, inplace=inplace)
            else:
                raise NotImplementedError

            modules.append(activation)

        super().__init__(*modules)


def sanakoyeu_et_al_2018_residual_block(channels: int) -> ResidualBlock:
    def elementwise(fn, inputs):
        if isinstance(inputs, Collection):
            return tuple([fn(input) for input in inputs])
        return fn(inputs)

    def same_size_padding(kernel_size):
        return elementwise(lambda x: (x - 1) // 2, kernel_size)

    in_channels = out_channels = channels
    kernel_size = 3
    padding = same_size_padding(kernel_size)
    instance_norm = True

    residual = nn.Sequential(
        nn.ReflectionPad2d(padding),
        SanakoyeuEtAl2018ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="valid",
            instance_norm=instance_norm,
            use_act=False,
        ),
        nn.ReflectionPad2d(padding),
        SanakoyeuEtAl2018ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="valid",
            instance_norm=instance_norm,
            use_act=False,
        ),
    )

    return ResidualBlock(residual)


def sanakoyeu_et_al_2018_transformer_encoder(
    in_channels: int = 3, gf_dim: int = 32, instance_norm: bool = True,
) -> pystiche.SequentialModule:
    modules = (
        nn.ReflectionPad2d(15),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=in_channels,
            out_channels=gf_dim,
            kernel_size=3,
            stride=1,
            padding="valid",
            instance_norm=instance_norm,
        ),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=gf_dim,
            out_channels=gf_dim,
            kernel_size=3,
            stride=2,
            padding="valid",
            instance_norm=instance_norm,
        ),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=gf_dim,
            out_channels=gf_dim * 2,
            kernel_size=3,
            stride=2,
            padding="valid",
            instance_norm=instance_norm,
        ),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=gf_dim * 2,
            out_channels=gf_dim * 4,
            kernel_size=3,
            stride=2,
            padding="valid",
            instance_norm=instance_norm,
        ),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=gf_dim * 4,
            out_channels=gf_dim * 8,
            kernel_size=3,
            stride=2,
            padding="valid",
            instance_norm=instance_norm,
        ),
    )
    return pystiche.enc.SequentialEncoder(*modules)


def sanakoyeu_et_al_2018_transformer_decoder(
    out_channels: int = 3,
    gf_dim: int = 32,
    instance_norm: bool = True,
    num_res_block: int = 9,
) -> pystiche.SequentialModule:

    modules = []
    for i in range(num_res_block):
        modules.append(sanakoyeu_et_al_2018_residual_block(gf_dim * 8))

    modules.append(
        SanakoyeuEtAl2018ConvTransponseBlock(
            in_channels=gf_dim * 8,
            out_channels=gf_dim * 8,
            kernel_size=3,
            stride=2,
            padding="same",
            instance_norm=instance_norm,
        )
    )
    modules.append(
        SanakoyeuEtAl2018ConvTransponseBlock(
            in_channels=gf_dim * 8,
            out_channels=gf_dim * 4,
            kernel_size=3,
            stride=2,
            padding="same",
            instance_norm=instance_norm,
        )
    )
    modules.append(
        SanakoyeuEtAl2018ConvTransponseBlock(
            in_channels=gf_dim * 4,
            out_channels=gf_dim * 2,
            kernel_size=3,
            stride=2,
            padding="same",
            instance_norm=instance_norm,
        )
    )
    modules.append(
        SanakoyeuEtAl2018ConvTransponseBlock(
            in_channels=gf_dim * 2,
            out_channels=gf_dim,
            kernel_size=3,
            stride=2,
            padding="same",
            instance_norm=instance_norm,
        )
    )
    modules.append(nn.ReflectionPad2d(3))
    modules.append(
        sanakoyeu_et_al_2018_conv(
            in_channels=gf_dim,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding="valid",
        )
    )
    return pystiche.SequentialModule(*modules)


class SanakoyeuEtAl2018Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = sanakoyeu_et_al_2018_transformer_decoder()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(input)) * 2 - 1


class SanakoyeuEtAl2018Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = sanakoyeu_et_al_2018_transformer_encoder()
        self.decoder = SanakoyeuEtAl2018Decoder()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(input))


class SanakoyeuEtAl2018Discriminator(pystiche.Module):
    def __init__(self, pred_module, *modules: nn.Module):
        self.pred_module = pred_module
        super().__init__(indexed_children=modules)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        pred = []
        for i, module in enumerate(self.children()):
            if isinstance(module, self.pred_module):
                x, scale_pred = module(x)
                pred.append(scale_pred)
            else:
                x = module(x)
        return pred


class SanakoyeuEtAl2018DiscriminatorPredBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel_size: Union[Tuple[int, int], int],
        pred_kernel_size: Union[Tuple[int, int], int],
        padding: str = "same",
        instance_norm: bool = True,
        act: str = "lrelu",
        inplace: bool = True,
    ):
        super().__init__()
        self.forwardBlock = SanakoyeuEtAl2018ConvBlock(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            instance_norm=instance_norm,
            act=act,
            inplace=inplace,
        )
        self.predConv = sanakoyeu_et_al_2018_conv(
            in_channels=output_channel,
            out_channels=1,
            kernel_size=pred_kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.forwardBlock(input)
        pred = self.predConv(output)
        return output, pred


def sanakoyeu_et_al_2018_discriminator(
    input_channels: int = 3,
    df_dim: int = 32,
    instance_norm: bool = True,
    inplace: bool = True,
) -> SanakoyeuEtAl2018Discriminator:

    modules = [
        SanakoyeuEtAl2018DiscriminatorPredBlock(
            input_channels,
            df_dim * 2,
            kernel_size=5,
            pred_kernel_size=5,
            instance_norm=instance_norm,
            inplace=inplace,
        ),
        SanakoyeuEtAl2018DiscriminatorPredBlock(
            df_dim * 2,
            df_dim * 2,
            kernel_size=5,
            pred_kernel_size=10,
            instance_norm=instance_norm,
            inplace=inplace,
        ),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=df_dim * 2,
            out_channels=df_dim * 4,
            kernel_size=5,
            stride=2,
            padding="same",
            instance_norm=instance_norm,
            act="lrelu",
            inplace=inplace,
        ),
        SanakoyeuEtAl2018DiscriminatorPredBlock(
            df_dim * 4,
            df_dim * 8,
            kernel_size=5,
            pred_kernel_size=10,
            instance_norm=instance_norm,
            inplace=inplace,
        ),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=df_dim * 8,
            out_channels=df_dim * 8,
            kernel_size=5,
            stride=2,
            padding="same",
            instance_norm=instance_norm,
            act="lrelu",
            inplace=inplace,
        ),
        SanakoyeuEtAl2018DiscriminatorPredBlock(
            df_dim * 8,
            df_dim * 16,
            kernel_size=5,
            pred_kernel_size=6,
            instance_norm=instance_norm,
            inplace=inplace,
        ),
        SanakoyeuEtAl2018DiscriminatorPredBlock(
            df_dim * 16,
            df_dim * 16,
            kernel_size=5,
            pred_kernel_size=3,
            instance_norm=instance_norm,
            inplace=inplace,
        ),
    ]
    return SanakoyeuEtAl2018Discriminator(
        SanakoyeuEtAl2018DiscriminatorPredBlock, *modules
    )


class SanakoyeuEtAl2018TransformerBlock(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        kernel_size: Union[Tuple[int, int], int] = 10,
        stride: Union[Tuple[int, int], int] = 1,
        padding: str = "same",
        impl_params: bool = True,
    ):
        super().__init__()
        self.impl_params = impl_params

        padding = get_padding(padding, kernel_size)

        if self.impl_params:
            self.forwardBlock = nn.AvgPool2d(
                kernel_size=kernel_size, stride=stride, padding=padding
            )
        else:
            self.forwardBlock = nn.Conv2d(
                input_channel, 3, kernel_size, stride=stride, padding=padding,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.impl_params:
            return self.forwardBlock(input)
        else:
            return nn.utils.weight_norm(self.forwardBlock(input))
