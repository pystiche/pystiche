from typing import cast

import torch
from torch import nn
from torch.nn.functional import interpolate

from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
    ExpiredCopyrightLicense,
    PixabayLicense,
    PublicDomainLicense,
)

__all__ = ["images", "transformer"]


def images() -> DownloadableImageCollection:
    """Collection of images used in the usage examples.

    .. note::

        You can use

        .. code-block:: python

            print(images())

        to get a list of all available images.
    """
    return DownloadableImageCollection(
        {
            "bird1": DownloadableImage(
                "https://download.pystiche.org/images/bird1.jpg",
                file="bird1.jpg",
                author="gholmz0",
                date="09.03.2013",
                license=PixabayLicense(),
                md5="ca869559701156c66fc1519ffc26123c",
                note="https://pixabay.com/photos/bird-wildlife-australian-bird-1139734/",
            ),
            "paint": DownloadableImage(
                "https://download.pystiche.org/images/paint.jpg",
                file="paint.jpg",
                author="garageband",
                date="03.07.2017",
                license=PixabayLicense(),
                md5="a991e222806ef49d34b172a67cf97d91",
                note="https://pixabay.com/de/photos/abstrakt-kunst-hintergrund-farbe-2468874/",
            ),
            "bird2": DownloadableImage(
                "https://download.pystiche.org/images/bird2.jpg",
                file="bird2.jpg",
                author="12019",
                date="09.04.2012",
                license=PixabayLicense(),
                md5="2398e8f6d102b9b8de7c74d4c73b71f4",
                note="https://pixabay.com/photos/bird-wildlife-sky-clouds-nature-92956/",
            ),
            "mosaic": DownloadableImage(
                "https://upload.wikimedia.org/wikipedia/commons/2/23/Mosaic_ducks_Massimo.jpg",
                file="mosaic.jpg",
                author="Marie-Lan Nguyen",
                date="2006",
                license=PublicDomainLicense(),
                md5="5b60cd1724395f7a0c21dc6dd006f8ae",
            ),
            "castle": DownloadableImage(
                "https://download.pystiche.org/images/castle.jpg",
                file="castle.jpg",
                author="Christian Hänsel",
                date="23.03.2014",
                license=PixabayLicense(),
                md5="9d085fac77065e0da66629162f759f90",
                note="https://pixabay.com/de/photos/lemgo-brake-schloss-br%C3%BCcke-310586/",
                guides=DownloadableImageCollection(
                    {
                        "building": DownloadableImage(
                            "https://download.pystiche.org/images/castle/building.png",
                            file="building.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="680bd875f4fdb919d446d6210cbe1035",
                        ),
                        "sky": DownloadableImage(
                            "https://download.pystiche.org/images/castle/sky.png",
                            file="sky.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="c4d5f18b3ca22a1ff879f40ca797ed26",
                        ),
                        "water": DownloadableImage(
                            "https://download.pystiche.org/images/castle/water.png",
                            file="water.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="2642552e8c98b6a63f24dd67bb54e2a3",
                        ),
                    },
                ),
            ),
            "church": DownloadableImage(
                "https://upload.wikimedia.org/wikipedia/commons/a/ad/Vincent_van_Gogh_-_The_Church_in_Auvers-sur-Oise%2C_View_from_the_Chevet_-_Google_Art_Project.jpg",
                file="church.jpg",
                author="Vincent van Gogh",
                title="The Church at Auvers",
                date="1890",
                license=ExpiredCopyrightLicense(1890),
                md5="fd866289498367afd72a5c9cf626073a",
                guides=DownloadableImageCollection(
                    {
                        "building": DownloadableImage(
                            "https://download.pystiche.org/images/church/building.png",
                            file="building.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="41256f62d2bd1560ed1e66623c8d9c9f",
                        ),
                        "sky": DownloadableImage(
                            "https://download.pystiche.org/images/church/sky.png",
                            file="sky.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="c87c7234d2788a1d2a4e1633723c794b",
                        ),
                        "surroundings": DownloadableImage(
                            "https://download.pystiche.org/images/church/surroundings.png",
                            file="surroundings.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="807a40beed727af81333edbd8eb89aff",
                        ),
                    }
                ),
            ),
            "cliff": DownloadableImage(
                "https://upload.wikimedia.org/wikipedia/commons/a/a4/Claude_Monet_-_Cliff_Walk_at_Pourville_-_Google_Art_Project.jpg",
                file="cliff.jpg",
                author="Claude Monet",
                title="The Cliff Walk at Pourville",
                date="1882",
                license=ExpiredCopyrightLicense(1926),
                md5="8f6b8101b484f17cea92a12ad27be31d",
                guides=DownloadableImageCollection(
                    {
                        "landscape": DownloadableImage(
                            "https://download.pystiche.org/images/cliff/landscape.png",
                            file="landscape.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="4d02ee8aaa610ced3ef12a5c82f29b81",
                        ),
                        "sky": DownloadableImage(
                            "https://download.pystiche.org/images/cliff/sky.png",
                            file="sky.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="8f76b7ed88ff1b23feaab1de51501ae1",
                        ),
                        "water": DownloadableImage(
                            "https://download.pystiche.org/images/cliff/water.png",
                            file="water.png",
                            author="Julian Bültemeier",
                            license=PublicDomainLicense(),
                            md5="f0cab02afadf38d3262ce1c61a36f6da",
                        ),
                    }
                ),
            ),
        }
    )


class Interpolate(nn.Module):
    def __init__(self, scale_factor: float = 1.0, mode: str = "nearest") -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            interpolate(input, scale_factor=self.scale_factor, mode=self.mode),
        )

    def extra_repr(self) -> str:
        extras = [f"scale_factor={self.scale_factor}"]
        if self.mode != "nearest":
            extras.append(f"mode={self.mode}")
        return ", ".join(extras)


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        upsample: bool = False,
        norm: bool = True,
        activation: bool = True,
    ):
        super().__init__()
        self.upsample = Interpolate(scale_factor=stride) if upsample else None
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1 if upsample else stride
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if norm else None
        self.activation = nn.ReLU() if activation else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            input = self.upsample(input)

        output = self.conv(self.pad(input))

        if self.norm:
            output = self.norm(output)
        if self.activation:
            output = self.activation(output)

        return cast(torch.Tensor, output)


class Residual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = Conv(channels, channels, kernel_size=3)
        self.conv2 = Conv(channels, channels, kernel_size=3, activation=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = cast(torch.Tensor, self.conv2(self.conv1(input)))
        return output + input


class FloatToUint8Range(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * 255.0


class Uint8ToFloatRange(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input / 255.0


class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            Conv(3, 32, kernel_size=9),
            Conv(32, 64, kernel_size=3, stride=2),
            Conv(64, 128, kernel_size=3, stride=2),
            Residual(128),
            Residual(128),
            Residual(128),
            Residual(128),
            Residual(128),
        )
        self.decoder = nn.Sequential(
            Conv(128, 64, kernel_size=3, stride=2, upsample=True),
            Conv(64, 32, kernel_size=3, stride=2, upsample=True),
            Conv(32, 3, kernel_size=9, norm=False, activation=False),
        )

        self.preprocessor = FloatToUint8Range()
        self.postprocessor = Uint8ToFloatRange()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.preprocessor(input)
        output = self.decoder(self.encoder(input))
        return cast(torch.Tensor, self.postprocessor(output))


def transformer() -> nn.Module:
    """Basic transformer for model-based optimization.

    The transformer is compatible with the
    `official PyTorch example <https://github.com/pytorch/examples/tree/master/fast_neural_style>`_
    which in turn is based on :cite:`JAL2016`
    """
    return Transformer()
