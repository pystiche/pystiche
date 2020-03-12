from typing import Optional, Sized
from urllib.parse import urljoin
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
    CreativeCommonsLicense,
    PublicDomainLicense,
    ImageFolderDataset,
    FiniteCycleBatchSampler,
)
from pystiche.image import extract_num_channels, CaffePreprocessing
from pystiche.image.transforms import Transform, Resize, ComposedTransform
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
    edge_size: int = 256, impl_params: bool = True, instance_norm: bool = True,
) -> ComposedTransform:
    class OptionalGrayscaleToFakegrayscale(Transform):
        def forward(self, input_image: torch.Tensor) -> torch.Tensor:
            is_grayscale = extract_num_channels(input_image) == 1
            if is_grayscale:
                return grayscale_to_fakegrayscale(input_image)
            else:
                return input_image

    transforms = []
    if impl_params:
        if instance_norm:
            # FIXME: RandomCrop here
            transforms.append(
                Resize((edge_size, edge_size), interpolation_mode="bicubic")
            )
        else:
            transforms.append(
                Resize((edge_size, edge_size), interpolation_mode="bilinear")
            )
    else:
        transforms.append(Resize((edge_size, edge_size), interpolation_mode="bilinear"))

    transforms.append(OptionalGrayscaleToFakegrayscale())
    return ComposedTransform(*transforms)


def ulyanov_et_al_2016_style_transform(
    impl_params: bool = True,
    instance_norm: bool = True,
    edge_size: Optional[int] = None,
) -> ComposedTransform:
    if edge_size is None:
        edge_size = 256

    if impl_params:
        interpolation_mode = "bicubic" if instance_norm else "bilinear"
    else:
        interpolation_mode = "bilinear"

    transforms = [
        Resize(edge_size, edge="long", interpolation_mode=interpolation_mode),
        CaffePreprocessing(),
    ]
    return ComposedTransform(*transforms)


def ulyanov_et_al_2016_images(
    root: Optional[str] = None, download: bool = True, overwrite: bool = False
):

    base_ulyanov = (
        "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/master/data/"
    )
    base_ulyanov_suppl = "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/texture_nets_v1/supplementary/"
    readme_ulyanov = "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/texture_nets_v1/data/readme_pics/"
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
        "bird": DownloadableImage(
            urljoin(base_ulyanov_suppl, "bird.jpg"),
            md5="74dde9fad4749e7ff3cd4eca6cb43d0d",
        ),
        "kitty": DownloadableImage(
            urljoin(readme_ulyanov, "kitty.jpg"),
            md5="98262bd8f5ae25f8329158d2c2c66ad0",
        ),
    }

    texture_base_ulyanov = urljoin(base_ulyanov, "textures/")
    base_ulyanov_suppl_texture = "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/texture_nets_v1/supplementary//texture_models/"

    texture_images = {
        "cezanne": DownloadableImage(
            urljoin(texture_base_ulyanov, "cezanne.jpg"),
            md5="fab6d360c361c38c331b3ee5ef0078f5",
        ),
        "bricks": DownloadableImage(
            urljoin(base_ulyanov_suppl_texture, "bricks.png"),
            md5="1e13818e1fbefbd22f110a1c2f781d40",
        ),
        "pebble": DownloadableImage(
            urljoin(base_ulyanov_suppl_texture, "pebble.png"),
            md5="5b5e5aa6c579e42e268058a94d683a6c",
        ),
        "pixelcity_windows2": DownloadableImage(
            urljoin(base_ulyanov_suppl_texture, "pixelcity_windows2.jpg"),
            md5="53026a8411e7c26e959e36d3223f3b8f",
        ),
        "red-peppers256.o": DownloadableImage(
            urljoin(base_ulyanov_suppl_texture, "red-peppers256.o.jpg"),
            md5="16371574a10e0d10b88b807204c4f546",
        ),
    }
    base_johnson = (
        "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/"
    )
    style_base_johnson = urljoin(base_johnson, "styles/")

    base_ulyanov_suppl_style = "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/texture_nets_v1/supplementary//stylization_models/"
    style_images = {
        "candy": DownloadableImage(
            urljoin(style_base_johnson, "candy.jpg"),
            md5="00a0e3aa9775546f98abf6417e3cb478",
        ),
        "the_scream": DownloadableImage(
            urljoin(style_base_johnson, "the_scream.jpg"),
            md5="619b4f42c84d2b62d3518fb20fa619c2",
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
        "mosaic": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "mosaic.jpg"),
            md5="4f05f1e12961cebf41bd372d909342b3",
        ),
        "pleades": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "pleades.jpg"),
            md5="6fc41ac30c2852a5454a0ead2f479dc9",
        ),
        "starry": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "starry.jpg"),
            md5="c6d94f7962466b2e80a64ae82523242a",
        ),
        "turner": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "turner.jpg"),
            md5="7fdd9603a5182dcef23d7fb1c5217888",
        ),
    }
    return DownloadableImageCollection(
        {**texture_images, **content_images, **style_images},
        root=root,
        download=download,
        overwrite=overwrite,
    )


def ulyanov_et_al_2016_dataset(
    root: str,
    impl_params: bool = True,
    instance_norm: bool = True,
    transform: Optional[Transform] = None,
):
    if transform is None:
        transform = ulyanov_et_al_2016_content_transform(
            impl_params=impl_params, instance_norm=instance_norm
        )
    return ImageFolderDataset(root, transform=transform)


def ulyanov_et_al_2016_batch_sampler(
    data_source: Sized,
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    num_batches=None,
    batch_size=None,
) -> FiniteCycleBatchSampler:

    if num_batches is None:
        if impl_params:
            if instance_norm:
                num_batches = 2000
            else:
                num_batches = 300 if stylization else 150
        else:
            num_batches = 200

    if batch_size is None:
        if impl_params:
            if instance_norm:
                batch_size = 1
            else:
                batch_size = 4 if stylization else 16
        else:
            batch_size = 16

    return FiniteCycleBatchSampler(
        data_source, num_batches=num_batches, batch_size=batch_size
    )


def ulyanov_et_al_2016_image_loader(
    dataset: Dataset,
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    batch_sampler: Optional[Sampler] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    if batch_sampler is None:
        batch_sampler = ulyanov_et_al_2016_batch_sampler(
            dataset,
            impl_params=impl_params,
            instance_norm=instance_norm,
            stylization=stylization,
        )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
