from torch import nn
from typing import Union, Tuple, Collection, Dict
from collections import OrderedDict
import pystiche
from pystiche.enc.encoder import SequentialEncoder
from pystiche.enc.multi_layer_encoder import MultiLayerEncoder
import torch
from ..common_utils import ResidualBlock


def get_padding(
    padding: str, kernel_size: Union[Tuple[int, int], int]
) -> int:  # TODO: move this to pystiche.? used in two papers
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

        modules.append(nn.InstanceNorm2d(out_channels))

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

        modules.append(nn.InstanceNorm2d(out_channels))

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

    residual = nn.Sequential(
        nn.ReflectionPad2d(padding),
        SanakoyeuEtAl2018ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="valid",
            use_act=False,
        ),
        nn.ReflectionPad2d(padding),
        SanakoyeuEtAl2018ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="valid",
            use_act=False,
        ),
    )

    return ResidualBlock(residual)


def sanakoyeu_et_al_2018_transformer_encoder(
    in_channels: int = 3,
) -> pystiche.SequentialModule:
    modules = (
        nn.ReflectionPad2d(15),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding="valid",
        ),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding="valid",
        ),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding="valid",
        ),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding="valid",
        ),
        SanakoyeuEtAl2018ConvBlock(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding="valid",
        ),
    )
    return SequentialEncoder(modules)


def sanakoyeu_et_al_2018_transformer_decoder(
    out_channels: int = 3, num_res_block: int = 9,
) -> pystiche.SequentialModule:

    modules = []
    for i in range(num_res_block):
        modules.append(sanakoyeu_et_al_2018_residual_block(256))

    modules.append(
        SanakoyeuEtAl2018ConvTransponseBlock(
            in_channels=256, out_channels=256, kernel_size=3, stride=2, padding="same",
        )
    )
    modules.append(
        SanakoyeuEtAl2018ConvTransponseBlock(
            in_channels=256, out_channels=128, kernel_size=3, stride=2, padding="same",
        )
    )
    modules.append(
        SanakoyeuEtAl2018ConvTransponseBlock(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding="same",
        )
    )
    modules.append(
        SanakoyeuEtAl2018ConvTransponseBlock(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, padding="same",
        )
    )
    modules.append(nn.ReflectionPad2d(3))
    modules.append(
        sanakoyeu_et_al_2018_conv(
            in_channels=32,
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


class SanakoyeuEtAl2018Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = sanakoyeu_et_al_2018_transformer_encoder()
        self.decoder = SanakoyeuEtAl2018Decoder()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(input))


class SanakoyeuEtAl2018TransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
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
                in_channels, 3, kernel_size, stride=stride, padding=padding,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.impl_params:
            return self.forwardBlock(input)
        else:
            return nn.utils.weight_norm(self.forwardBlock(input))


def sanakoyeu_et_al_2018_discriminator_encoder_modules(
    in_channels: int = 3, inplace: bool = True,
) -> Dict[str, nn.Sequential]:

    modules = OrderedDict(
        {
            "scale_0": SanakoyeuEtAl2018ConvBlock(
                in_channels,
                128,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_1": SanakoyeuEtAl2018ConvBlock(
                128,
                128,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_2": SanakoyeuEtAl2018ConvBlock(
                128,
                256,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_3": SanakoyeuEtAl2018ConvBlock(
                256,
                512,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_4": SanakoyeuEtAl2018ConvBlock(
                in_channels=512,
                out_channels=512,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_5": SanakoyeuEtAl2018ConvBlock(
                512,
                1024,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_6": SanakoyeuEtAl2018ConvBlock(
                1024,
                1024,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
        }
    )
    return modules


def sanakoyeu_et_al_2018_prediction_module(
    in_channels: int, kernel_size: Union[Tuple[int, int], int], padding: str = "same"
):
    return sanakoyeu_et_al_2018_conv(
        in_channels=in_channels,
        out_channels=1,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )


class SanakoyeuEtAl2018DiscriminatorEncoder(MultiLayerEncoder):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__(
            self._extract_modules(
                sanakoyeu_et_al_2018_discriminator_encoder_modules(
                    in_channels=in_channels
                )
            )
        )

    def _extract_modules(self, wrapped_modules: Dict[str, nn.Sequential]):
        modules = OrderedDict()
        block = 0
        for sequential in wrapped_modules.values():
            for module in sequential._modules.values():
                if isinstance(module, nn.Conv2d):
                    name = f"conv{block}"
                elif isinstance(module, nn.InstanceNorm2d):
                    name = f"inst_n{block}"
                else:  # isinstance(module, nn.LeakyReLU):
                    name = f"lrelu{block}"
                    # each LeakyReLU layer marks the end of the current block
                    block += 1

                modules[name] = module
        return modules


class SanakoyeuEtAl2018Discriminator(object):
    def __init__(self, in_channels: int = 3) -> None:
        self.multi_layer_encoder = SanakoyeuEtAl2018DiscriminatorEncoder(
            in_channels=in_channels
        )
        self.prediction_modules = OrderedDict(
            {
                "lrelu0": sanakoyeu_et_al_2018_prediction_module(128, 5),
                "lrelu1": sanakoyeu_et_al_2018_prediction_module(128, 10),
                "lrelu3": sanakoyeu_et_al_2018_prediction_module(512, 10),
                "lrelu5": sanakoyeu_et_al_2018_prediction_module(1024, 6),
                "lrelu6": sanakoyeu_et_al_2018_prediction_module(1024, 3),
            }
        )

    def get_prediction_module(self, layer: str):
        return self.prediction_modules[layer]

    def get_discriminator_parameters(self):
        parameters = list(self.multi_layer_encoder.parameters())
        for module in self.prediction_modules.values():
            parameters += list(module.parameters())
        return parameters
