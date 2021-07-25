import os
from typing import Any, Callable, Iterator, List, Optional, Tuple, cast

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file

from pystiche.image import read_image

__all__ = [
    "ImageFolderDataset",
]


def walkupto(
    top: str, depth: Optional[int] = None, **kwargs: Any
) -> Iterator[Tuple[str, List[str], List[str]]]:
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
        transform: Optional[nn.Module] = None,
        depth: Optional[int] = None,
        importer: Optional[Callable[[str], Any]] = None,
    ):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.image_files = self._collect_image_files(depth)
        self.transform = transform

        if importer is None:

            def importer(file: str) -> torch.Tensor:
                return read_image(file, make_batched=False)

        self.importer = cast(Callable[[str], Any], importer)

    def _collect_image_files(self, depth: Optional[int]) -> Tuple[str, ...]:
        image_files = tuple(
            os.path.join(root, file)
            for root, _, files in walkupto(self.root, depth=depth)
            for file in files
            if is_image_file(file)
        )

        if not image_files:
            msg = f"The directory {self.root} does not contain any image files"
            if depth is not None:
                msg += f" up to a depth of {depth}"
            msg += "."
            raise RuntimeError(msg)

        return image_files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Any:
        file = self.image_files[idx]
        image = self.importer(file)

        if self.transform:
            image = self.transform(image)

        return image
