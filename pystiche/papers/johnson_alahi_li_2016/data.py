from typing import Optional, Sized, Tuple
from urllib.parse import urljoin

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
    FiniteCycleBatchSampler,
    ImageFolderDataset,
)
from pystiche.image import CaffePreprocessing, extract_image_size, extract_num_channels
from pystiche.image.transforms import ComposedTransform, Resize, Transform
from pystiche.image.transforms.functional import (
    grayscale_to_fakegrayscale,
    top_left_crop,
)

__all__ = [
    "johnson_alahi_li_2016_content_transform",
    "johnson_alahi_li_2016_style_transform",
    "johnson_alahi_li_2016_images",
    "johnson_alahi_li_2016_dataset",
    "johnson_alahi_li_2016_batch_sampler",
    "johnson_alahi_li_2016_image_loader",
]


def johnson_alahi_li_2016_content_transform(
    edge_size: int = 256, multiple: int = 16, impl_params: bool = True,
) -> ComposedTransform:
    class TopLeftCropToMultiple(Transform):
        def __init__(self, multiple: int):
            super().__init__()
            self.multiple = multiple

        def calculate_size(self, image: torch.Tensor) -> Tuple[int, int]:
            old_height, old_width = extract_image_size(image)
            new_height = old_height - old_height % self.multiple
            new_width = old_width - old_width % self.multiple
            return new_height, new_width

        def forward(self, image: torch.tensor) -> torch.Tensor:
            size = self.calculate_size(image)
            return top_left_crop(image, size)

    class OptionalGrayscaleToFakegrayscale(Transform):
        def forward(self, input_image: torch.Tensor) -> torch.Tensor:
            is_grayscale = extract_num_channels(input_image) == 1
            if is_grayscale:
                return grayscale_to_fakegrayscale(input_image)
            else:
                return input_image

    transforms = [
        TopLeftCropToMultiple(multiple),
        Resize((edge_size, edge_size)),
        OptionalGrayscaleToFakegrayscale(),
    ]
    if impl_params:
        transforms.append(CaffePreprocessing())

    return ComposedTransform(*transforms)


def get_style_edge_size(
    impl_params: bool, instance_norm: bool, style: Optional[str] = None
) -> int:
    def get_default_edge_size():
        if not impl_params and not instance_norm:
            return 256
        elif impl_params and not instance_norm:
            return 512
        elif impl_params and instance_norm:
            return 384
        else:
            raise RuntimeError

    default_edge_size = get_default_edge_size()

    if style is None or not instance_norm:
        return default_edge_size

    edge_sizes = {
        "candy": 384,
        "la_muse": 512,
        "mosaic": 512,
        "feathers": 180,
        "the_scream": 384,
        "udnie": 256,
    }
    try:
        return edge_sizes[style]
    except KeyError:
        return default_edge_size


def johnson_alahi_li_2016_style_transform(
    impl_params: bool = True,
    instance_norm: bool = True,
    style: Optional[str] = None,
    edge_size: Optional[int] = None,
    edge="long",
) -> Resize:
    if edge_size is None:
        edge_size = get_style_edge_size(impl_params, instance_norm, style=style)
    return Resize(edge_size, edge=edge)


def johnson_alahi_li_2016_images(
    root: Optional[str] = None, download: bool = True, overwrite: bool = False
):
    base = (
        "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/"
    )

    content_base = urljoin(base, "content/")
    content_images = {
        "chicago": DownloadableImage(
            urljoin(content_base, "chicago.jpg"), md5="16ea186230a8a5131b224ddde01d0dd5"
        ),
        "hoovertowernight": DownloadableImage(
            urljoin(content_base, "hoovertowernight.jpg"),
            md5="97f7bab04e1f4c852fd2499356163b15",
        ),
    }

    style_base = urljoin(base, "styles/")
    style_images = {
        "candy": DownloadableImage(
            urljoin(style_base, "candy.jpg"), md5="00a0e3aa9775546f98abf6417e3cb478"
        ),
        "composition_vii": DownloadableImage(
            urljoin(style_base, "composition_vii.jpg"),
            md5="8d4f97cb0e8b1b07dee923599ee86cbd",
        ),
        "feathers": DownloadableImage(
            urljoin(style_base, "feathers.jpg"), md5="461c8a1704b59af1cf686883b16feec6"
        ),
        "la_muse": DownloadableImage(
            urljoin(style_base, "la_muse.jpg"), md5="77262ef6985cc427f84d78784ab5c1d8"
        ),
        "mosaic": DownloadableImage(
            urljoin(style_base, "mosaic.jpg"), md5="67b11e9cb1a69df08d70d9c2c7778fba"
        ),
        "starry_night": DownloadableImage(
            urljoin(style_base, "starry_night.jpg"),
            md5="ff217acb6db32785b8651a0e316aeab3",
        ),
        "the_scream": DownloadableImage(
            urljoin(style_base, "the_scream.jpg"),
            md5="619b4f42c84d2b62d3518fb20fa619c2",
        ),
        "udnie": DownloadableImage(
            urljoin(style_base, "udnie.jpg"), md5="6f3fa51706b21580a4b77f232d3b8ba9"
        ),
        "the_wave": DownloadableImage(
            urljoin(style_base, "wave.jpg"),
            md5="b06acee16641a2a04fb87bade8cee529",
            file="the_wave.jpg",
        ),
    }
    return DownloadableImageCollection(
        {**content_images, **style_images},
        root=root,
        download=download,
        overwrite=overwrite,
    )


def johnson_alahi_li_2016_dataset(
    root: str, impl_params: bool = True, transform: Optional[Transform] = None,
):
    if transform is None:
        transform = johnson_alahi_li_2016_content_transform(impl_params=impl_params)

    return ImageFolderDataset(root, transform=transform)


def johnson_alahi_li_2016_batch_sampler(
    data_source: Sized, num_batches=40000, batch_size=4
) -> FiniteCycleBatchSampler:
    return FiniteCycleBatchSampler(
        data_source, num_batches=num_batches, batch_size=batch_size
    )


def johnson_alahi_li_2016_image_loader(
    dataset: Dataset,
    batch_sampler: Optional[Sampler] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    if batch_sampler is None:
        batch_sampler = johnson_alahi_li_2016_batch_sampler(dataset)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
