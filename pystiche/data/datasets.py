import os
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import is_image_file

from pystiche.image import read_image

__all__ = [
    "Unsupervised",
    "ImageFolderDataset",
]


class Unsupervised:
    def __init__(self, *args, **kwargs):
        if not isinstance(self, VisionDataset):
            raise RuntimeError
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        return super().__getitem__(index)[0]


def walkupto(top: str, depth: Optional[int] = None, **kwargs: Any):
    if depth is None:
        yield from os.walk(top, **kwargs)
        return

    base_depth = top.count(os.sep)
    for root, dirs, files in os.walk(top, **kwargs):
        if root.count(os.sep) <= base_depth + depth:
            yield root, dirs, files
            # FIXME: stop walking directories if top directory is already to deep


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: [nn.Module] = None,
        depth: Optional[int] = None,
        importer: Optional[Callable[[str], torch.Tensor]] = None,
    ):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.image_files = self._collect_image_files(depth)
        self.transform = transform

        if importer is None:

            def importer(file: str) -> torch.Tensor:
                return read_image(file, make_batched=False)

        self.importer = importer

    def _collect_image_files(self, depth: Optional[int]):
        image_files = tuple(
            [
                os.path.join(root, file)
                for root, _, files in walkupto(self.root, depth=depth)
                for file in files
                if is_image_file(file)
            ]
        )
        if len(image_files) == 0:
            msg = f"The directory {self.root} does not contain any image files"
            if depth is not None:
                msg += f" up to a depth of {depth}"
            msg += "."
            raise RuntimeError(msg)

        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        file = self.image_files[idx]
        image = self.importer(file)

        if self.transform:
            image = self.transform(image)

        return image
