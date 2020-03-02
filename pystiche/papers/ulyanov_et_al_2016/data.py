from typing import Optional, Sized
from urllib.parse import urljoin
import torch
from torchvision.transforms import RandomCrop
from torch.utils.data import Dataset, Sampler, DataLoader
from pystiche.image import extract_num_channels, CaffePreprocessing
from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
    CreativeCommonsLicense,
    PublicDomainLicense,
    ImageFolderDataset,
    FiniteCycleBatchSampler,
)
from pystiche.image.transforms import Transform, ComposedTransform, Rescale, Resize
from pystiche.image.transforms.functional import grayscale_to_fakegrayscale


__all__ = [
    "ulyanov_et_al_2016_content_transform",
    "ulyanov_et_al_2016_style_transform",
    "ulyanov_et_al_2016_images",
    "ulyanov_et_al_2016_dataset",
    "ulyanov_et_al_2016_batch_sampler",
    "ulyanov_et_al_2016_image_loader",
]


def ulyanov_et_al_2016_content_transform(
    edge_size: int = 256, impl_params: bool = True,
) -> ComposedTransform:
    class OptionalGrayscaleToFakegrayscale(Transform):
        def forward(self, input_image: torch.Tensor) -> torch.Tensor:
            is_grayscale = extract_num_channels(input_image) == 1
            if is_grayscale:
                return grayscale_to_fakegrayscale(input_image)
            else:
                return input_image

    transforms = [
        # RandomCrop((edge_size, edge_size)),  # FIXME: check this
        Resize((edge_size, edge_size)),
        OptionalGrayscaleToFakegrayscale(),
    ]
    if impl_params:
        transforms.append(CaffePreprocessing())

    return ComposedTransform(*transforms)


def ulyanov_et_al_2016_style_transform(
    impl_params: bool = True, edge_size: Optional[int] = None,
) -> Rescale:
    if edge_size is None:
        edge_size = 256 if impl_params else 512
    return Resize((edge_size, edge_size), interpolation_mode="bicubic")


def ulyanov_et_al_2016_images(
    root: Optional[str] = None, download: bool = True, overwrite: bool = False
):

    base_johnson = (
        "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/"
    )

    base_ulyanov = (
        "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/master/data/"
    )
    # FIXME md5
    content_base_johnson = urljoin(base_johnson, "styles/")
    content_base_ulyanov = urljoin(base_ulyanov, "readme_pics/")
    content_images = {
        "karya": DownloadableImage(
            urljoin(content_base_ulyanov, "karya.jpg"),
            md5="232b2f03a5d20c453a41a0e6320f27be",
        ),
        "tiger": DownloadableImage(
            urljoin(content_base_ulyanov, "tiger.jpg"),
            md5="e82bf374da425fb2c2e2a35a5a751989",
        ),
        "neckarfront": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg",
            title="Tübingen Neckarfront",
            author="Andreas Praefcke",
            license=CreativeCommonsLicense(("by",), version="3.0"),
            md5="dc9ad203263f34352e18bc29b03e1066",
            file="tuebingen_neckarfront__andreas_praefcke.jpg",
        ),
        "Lena": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
            title="Lenna (test_image)",
            author="Dwight Hooker",
            date="1972",
            license="TODO",  # TODO
            md5="814a0034f5549e957ee61360d87457e5",
            file="Lenna_(test_image).png",
        ),
        "CheHigh": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/5/58/CheHigh.jpg",
            title="CheHigh",
            author="Alberto Korda",
            date="1960",
            license=PublicDomainLicense(1960),
            md5="cffc0768090c5a705cbb30fdc24c3e64",
            file="CheHigh.jpg",
        ),
        "The_Tower_of_Babel": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/f/fc/Pieter_Bruegel_the_Elder_-_The_Tower_of_Babel_%28Vienna%29_-_Google_Art_Project_-_edited.jpg",
            title="The Tower of Babel",
            author="Pieter Bruegel",
            date="1563",
            license=PublicDomainLicense(1563),
            md5="1e113716c8aad6c2ca826ae0b83ffc76",
            file="the_tower_of_babel.jpg",
        ),
    }

    texture_base_ulyanov = urljoin(base_ulyanov, "textures/")
    # TODO: "https://www.cns.nyu.edu/~eero/texture/index.php#examples" license
    texture_base_simoncelli = "http://www.texturesynthesis.com/textures/Simoncelli/"
    texture_images = {
        "cezanne": DownloadableImage(
            urljoin(texture_base_ulyanov, "cezanne.jpg"),
            md5="fab6d360c361c38c331b3ee5ef0078f5",
        ),
        "red-peppers256.o": DownloadableImage(  # FIXME: MD5 problem
            urljoin(texture_base_simoncelli, "red-peppers256.o.jpg"),
            md5="16371574a10e0d10b88b807204c4f546",
        ),
        "g1_0747.o": DownloadableImage(  # FIXME: MD5 problem
            urljoin(texture_base_simoncelli, "g1_0747.o.jpg"),
            md5="25da69021ba99c81553e03c7956e68de",
        ),
        "d30_2076.o": DownloadableImage(
            urljoin(texture_base_simoncelli, "d30_2076.o.jpg"),
            md5="1ddbaa6815b7056c65a9bf5a4df9e0eb",
        ),
        "windowsP256.o": DownloadableImage(
            urljoin(texture_base_simoncelli, "windowsP256.o.jpg"),
            md5="cc6bb3819e0a392eb3a6a0eae60540db",
        ),
        "radishes256.o": DownloadableImage(
            urljoin(texture_base_simoncelli, "radishes256.o.jpg"),
            md5="243c8d8879db9730bc5cc741437dfa6c",
        ),
    }

    style_base_johnson = urljoin(base_johnson, "styles/")
    style_images = {
        "candy": DownloadableImage(
            urljoin(style_base_johnson, "candy.jpg"),
            md5="00a0e3aa9775546f98abf6417e3cb478",
        ),
        "starry_night": DownloadableImage(
            urljoin(style_base_johnson, "starry_night.jpg"),
            md5="ff217acb6db32785b8651a0e316aeab3",
        ),
        "the_scream": DownloadableImage(
            urljoin(style_base_johnson, "the_scream.jpg"),
            md5="619b4f42c84d2b62d3518fb20fa619c2",
        ),
        "shipwreck": DownloadableImage(
            "https://blog-imgs-51.fc2.com/b/e/l/bell1976brain/800px-Shipwreck_turner.jpg",
            title="Shipwreck of the Minotaur",
            author="J. M. W. Turner",
            date="ca. 1810",
            license=PublicDomainLicense(1851),
            md5="4fb76d6f6fc1678cb74e858324d4d0cb",
            file="shipwreck_of_the_minotaur__turner.jpg",
        ),
        "Mosaic_ducks_Massimo": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/2/23/Mosaic_ducks_Massimo.jpg",
            title="Mosaic ducks Massimo",
            author="Marie-Lan Nguyen",
            date="2006",
            license=PublicDomainLicense(2006),
            md5="5b60cd1724395f7a0c21dc6dd006f8ae",
            file="mosaic_ducks_massimo__nguyen.jpg",
        ),
        "Jean Metzinger": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/c/c9/Robert_Delaunay%2C_1906%2C_Portrait_de_Metzinger%2C_oil_on_canvas%2C_55_x_43_cm%2C_DSC08255.jpg",
            title="Portrait of Jean Metzinger",
            author="Jean Metzinger",
            date="1906",
            license=PublicDomainLicense(1906),
            md5="3539d50d2808b8eec5b05f892d8cf1e1",
            file="jean_metzinger.jpg",
        ),
    }
    return DownloadableImageCollection(
        {**texture_images, **content_images, **style_images},
        root=root,
        download=download,
        overwrite=overwrite,
    )


def ulyanov_et_al_2016_dataset(
    root: str, impl_params: bool = True, transform: Optional[Transform] = None,
):
    if transform is None:
        transform = ulyanov_et_al_2016_content_transform(impl_params=impl_params)

    return ImageFolderDataset(root, transform=transform)


def ulyanov_et_al_2016_batch_sampler(  # FIXME:batch_size=16 see also utils learning rate
    data_source: Sized, impl_params: bool = True, num_batches=2000, batch_size=4
) -> FiniteCycleBatchSampler:
    num_batches = 50000 if impl_params else num_batches
    batch_size = 1 if impl_params else batch_size

    return FiniteCycleBatchSampler(
        data_source, num_batches=num_batches, batch_size=batch_size
    )


def ulyanov_et_al_2016_image_loader(
    dataset: Dataset,
    impl_params: bool = True,
    batch_sampler: Optional[Sampler] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    if batch_sampler is None:
        batch_sampler = ulyanov_et_al_2016_batch_sampler(
            dataset, impl_params=impl_params
        )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
