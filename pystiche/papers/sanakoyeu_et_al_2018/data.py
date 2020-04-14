from typing import Optional, Sized, Tuple, List, Iterator, Dict, Any, Union
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from pystiche.data import (
    ImageFolderDataset,
    FiniteCycleBatchSampler,
    InfiniteCycleBatchSampler,
)
from pystiche.image.transforms import (
    Transform,
    ComposedTransform,
    ValidRandomCrop,
)
from pystiche.image import extract_num_channels
from pystiche.image.transforms.functional import grayscale_to_fakegrayscale
from pystiche.image.transforms import functional as F


class FiniteChangingCycleBatchSampler(InfiniteCycleBatchSampler):
    def __init__(
        self, data_source: Sized, num_batches: List[int], batch_size: int = 1,
    ):
        super().__init__(data_source, batch_size=batch_size)
        self.num_batches = num_batches
        self.pos = 0

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        iterator = super().__iter__()
        for _ in range(self.num_batches[self.pos]):
            yield next(iterator)
        if self.pos < len(self.num_batches) - 1:
            self.pos += 1

    def __len__(self) -> int:
        return self.num_batches[self.pos]



def sanakoyeu_et_al_2018_image_transform(
    edge_size: int = 768, impl_params: bool = True
) -> ComposedTransform:
    class OptionalRescale(Transform):
        def __init__(
                self, interpolation_mode: str = "bilinear",
        ):
            super().__init__()
            self.interpolation_mode = interpolation_mode

        def forward(
                self, image: torch.Tensor, max_size: float = 1800.0, min_size: float = 800.0
        ) -> torch.Tensor:
            if max(image.shape) > 1800.0:
                return F.rescale(image, factor=1800.0 / max(image.shape))
            if min(image.shape) < 800:
                alpha = 800.0 / float(min(image.shape))
                if alpha < 4.0:
                    return F.rescale(image, factor=alpha)
                else:
                    return F.resize(image, (800, 800), self.interpolation_mode)
            return image

        def _properties(self) -> Dict[str, Any]:
            dct = super()._properties()
            if self.interpolation_mode != "bilinear":
                dct["interpolation_mode"] = self.interpolation_mode
            return dct


    class OptionalGrayscaleToFakegrayscale(Transform):
        def forward(self, input_image: torch.Tensor) -> torch.Tensor:
            is_grayscale = extract_num_channels(input_image) == 1
            if is_grayscale:
                return grayscale_to_fakegrayscale(input_image)
            else:
                return input_image

    transforms = []
    if impl_params:
        transforms.append(OptionalRescale())
    transforms.append(ValidRandomCrop((edge_size, edge_size)))
    transforms.append(OptionalGrayscaleToFakegrayscale())
    return ComposedTransform(*transforms)


def sanakoyeu_et_al_2018_images(
    root: Optional[str] = None, download: bool = True, overwrite: bool = False
):
    # base_sanakoyeu = "https://hcicloud.iwr.uni-heidelberg.de/index.php/s/NcJj2oLBTYuT1tf/download?path=%2F&files="
    # tar1 = "berthe-morisot.tar.gz"
    # tar2 = "edvard-munch.tar.gz"
    # tar3 = "el-greco.tar.gz"
    # tar4 = "ernst-ludwig-kirchner.tar.gz"
    # tar5 = "jackson-pollock.tar.gz"
    # tar6 = "monet_water-lilies-1914.tar.gz"
    # tar7 = "nicholas-roerich.tar.gz"
    # tar8 = "pablo-picasso.tar.gz"
    # tar9 = "paul-cezanne.tar.gz"
    # tar10 = "paul-gauguin.tar.gz"
    # tar11 = "sample_photographs.tar.gz"
    # tar12 = "samuel-peploe.tar.gz"
    # tar13 = "vincent-van-gogh_road-with-cypresses-1890.tar.gz"
    # tar14 = "wassily-kandinsky.tar.gz"
    # places365_url = (
    #     "data.csail.mit.edu/places/places365/train_large_places365standard.tar"
    # )


    return None


def sanakoyeu_et_al_2018_dataset(
    root: str, impl_params: bool = True, transform: Optional[Transform] = None,
):
    if transform is None:
        transform = sanakoyeu_et_al_2018_image_transform(impl_params=impl_params)
    return ImageFolderDataset(root, transform=transform)


def sanakoyeu_et_al_2018_batch_sampler(
    data_source: Sized,
    impl_params: bool = True,
    num_batches: Union[int, List[int]] = None,
    batch_size: Optional[int] = None,
) -> Union[FiniteCycleBatchSampler, FiniteChangingCycleBatchSampler]:

    if num_batches is None:
        num_batches = int(3e5) if impl_params else [int(2e5), int(1e5)]

    if batch_size is None:
        batch_size = 1

    if isinstance(num_batches, int):
        return FiniteCycleBatchSampler(
            data_source, num_batches=num_batches, batch_size=batch_size
        )
    else:
        return FiniteChangingCycleBatchSampler(
            data_source, num_batches=num_batches, batch_size=batch_size
        )


def sanakoyeu_et_al_2018_image_loader(
    dataset: Dataset,
    impl_params: bool = True,
    batch_sampler: Optional[Sampler] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    if batch_sampler is None:
        batch_sampler = sanakoyeu_et_al_2018_batch_sampler(
            dataset, impl_params=impl_params
        )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
