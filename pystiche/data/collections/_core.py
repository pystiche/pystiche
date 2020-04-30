from os import path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch import nn

import pystiche
from pystiche.image import read_image


class _Image(pystiche.ComplexObject):
    def __init__(
        self,
        file: str,
        guides: Optional["_ImageCollection"] = None,
        transform: Optional[nn.Module] = None,
        note: Optional[str] = None,
    ):
        self.file = file
        self.guides = guides
        self.transform = transform
        self.note = note

    def read(
        self, root: Optional[str] = None, **read_image_kwargs: Any,
    ) -> torch.Tensor:
        file = self.file
        if not path.isabs(file) and root is not None:
            file = path.join(root, file)

        image = read_image(file, **read_image_kwargs)
        if self.transform is None:
            return image

        return self.transform(image)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["file"] = self.file
        if self.transform is not None:
            dct["transform"] = self.transform
        if self.note is not None:
            dct["note"] = self.note
        return dct

    def _named_children(self) -> Iterator[Tuple[str, Any]]:
        yield from super()._named_children()
        if self.guides is not None:
            yield from self.guides


class _ImageCollection(pystiche.ComplexObject):
    def __init__(self, images: Dict[str, _Image]):
        self._images = images

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, item) -> _Image:
        return self._images[item]

    def __iter__(self) -> Iterator[Tuple[str, _Image]]:
        for name, image in self._images.items():
            yield name, image

    def read(
        self, root: Optional[str] = None, **read_image_kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        return {
            name: image.read(root=root, **read_image_kwargs) for name, image in self
        }

    def _named_children(self) -> Iterator[Tuple[str, Any]]:
        yield from iter(self._images.items())
