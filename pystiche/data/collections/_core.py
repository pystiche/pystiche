from os import path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch
from torch import nn

import pystiche
from pystiche.image import read_guides, read_image


class _Image(pystiche.ComplexObject):
    def __init__(
        self,
        file: str,
        transform: Optional[nn.Module] = None,
        note: Optional[str] = None,
    ):
        self.file = file
        self.transform = transform
        self.note = note

    def read(
        self, root: Optional[str] = None, **read_image_kwargs: Any,
    ) -> torch.Tensor:
        if root is None:
            file = self.file
        else:
            file = path.join(root, path.basename(self.file))

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


class _ImageCollection(pystiche.ComplexObject):
    def __init__(self, images: Dict[str, _Image]):
        self._images = images

    def _named_children(self) -> Iterator[Tuple[str, Any]]:
        yield from iter(self._images.items())

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, item):
        return self._images[item]
