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
    edge_size: int = 256, impl_params: bool = True, instance_norm: bool = False,
) -> ComposedTransform:
    class OptionalGrayscaleToFakegrayscale(Transform):
        def forward(self, input_image: torch.Tensor) -> torch.Tensor:
            is_grayscale = extract_num_channels(input_image) == 1
            if is_grayscale:
                return grayscale_to_fakegrayscale(input_image)
            else:
                return input_image

    transforms = []
    if instance_norm:
        # FIXME: RandomCrop here
        transforms.append(Resize((edge_size, edge_size), interpolation_mode="bicubic"))
    else:
        if impl_params:
            transforms.append(
                Resize((edge_size, edge_size), interpolation_mode="bilinear")
            )
        else:
            transforms.append(
                Resize((edge_size, edge_size), interpolation_mode="bilinear")
            )  # FIXME: paper?
    transforms.append(OptionalGrayscaleToFakegrayscale())
    return ComposedTransform(*transforms)


def ulyanov_et_al_2016_style_transform(
    impl_params: bool = True,
    instance_norm: bool = False,
    edge_size: Optional[int] = None,
) -> ComposedTransform:
    if edge_size is None:
        if instance_norm:
            edge_size = 256
        else:
            edge_size = 256 if impl_params else 256

    if instance_norm:
        interpolation_mode = "bicubic"
    else:
        interpolation_mode = "bilinear" if impl_params else "bicubic"  # FIXME: paper?

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
        # "bird": DownloadableImage(urljoin(base_ulyanov_suppl, "bird.jpg"), md5="",),
    }

    texture_base_ulyanov = urljoin(base_ulyanov, "textures/")
    base_ulyanov_suppl_texture = "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/texture_nets_v1/supplementary//texture_models/"
    # TODO: "https://www.cns.nyu.edu/~eero/texture/index.php#examples" license
    texture_base_simoncelli = "http://www.texturesynthesis.com/textures/Simoncelli/"
    texture_images = {
        "cezanne": DownloadableImage(
            urljoin(texture_base_ulyanov, "cezanne.jpg"),
            md5="fab6d360c361c38c331b3ee5ef0078f5",
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
        # "bricks": DownloadableImage(
        #     urljoin(base_ulyanov_suppl_texture, "bricks.png"), md5="",
        # ),
        # "pebble": DownloadableImage(
        #     urljoin(base_ulyanov_suppl_texture, "pebble.png"), md5="",
        # ),
        # "pixelcity_windows2": DownloadableImage(
        #     urljoin(base_ulyanov_suppl_texture, "pixelcity_windows2.jpg"), md5="",
        # ),
        # "red-peppers256.o": DownloadableImage(
        #     urljoin(base_ulyanov_suppl_texture, "red-peppers256.o.jpg"), md5="",
        # ),
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
        # "mosaic": DownloadableImage(
        #     urljoin(base_ulyanov_suppl_style, "mosaic.jpg"), md5="",
        # ),
        # "pleades": DownloadableImage(
        #     urljoin(base_ulyanov_suppl_style, "pleades.jpg"), md5="",
        # ),
        # "starry": DownloadableImage(
        #     urljoin(base_ulyanov_suppl_style, "starry.jpg"), md5="",
        # ),
        # "turner": DownloadableImage(
        #     urljoin(base_ulyanov_suppl_style, "turner.jpg"), md5="",
        # ),
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
    instance_norm: bool = False,
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
    instance_norm: bool = False,
    mode: str = "texture",
    num_batches=None,
    batch_size=None,
) -> FiniteCycleBatchSampler:
    if num_batches is None:
        if instance_norm:
            num_batches = 50000
        else:
            if impl_params:
                num_batches = 3000 if mode == "style" else 1500
            else:
                num_batches = 2000

    if batch_size is None:
        if instance_norm:
            batch_size = 1
        else:
            if impl_params:
                batch_size = (
                    4 if mode == "style" else 4
                )  # FIXME change to 16 second entry
            else:
                batch_size = 4  # FIXME change to 16

    return FiniteCycleBatchSampler(
        data_source, num_batches=num_batches, batch_size=batch_size
    )


def ulyanov_et_al_2016_image_loader(
    dataset: Dataset,
    impl_params: bool = True,
    instance_norm: bool = False,
    mode: str = "texture",
    batch_sampler: Optional[Sampler] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    if batch_sampler is None:
        batch_sampler = ulyanov_et_al_2016_batch_sampler(
            dataset, impl_params=impl_params, instance_norm=instance_norm, mode=mode
        )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
