from os import path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple, TypeVar

import torch

import pystiche
from pystiche.image import read_image

_ImageCollection_co = TypeVar("_ImageCollection_co", covariant=True)


class _Image(pystiche.ComplexObject):
    def __init__(
        self,
        file: str,
        guides: Optional["_ImageCollection"] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        note: Optional[str] = None,
    ) -> None:
        self.file = file
        self.guides = guides
        self.transform = transform
        self.note = note

    def read(
        self, root: Optional[str] = None, **read_image_kwargs: Any,
    ) -> torch.Tensor:
        r"""Read the image from file with :func:`pystiche.image.read_image` and
        optionally apply :attr:`.transform`.

        Args:
            root: Optional root directory if the file is a relative path.
            **read_image_kwargs: Optional parameters passed to
                :func:`pystiche.image.read_image`.
        """
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
    def __init__(self, images: Mapping[str, _Image]) -> None:
        self._images = images

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, name: str) -> _Image:
        return self._images[name]

    def __iter__(self) -> Iterator[Tuple[str, _Image]]:
        for name, image in self._images.items():
            yield name, image

    def read(
        self, root: Optional[str] = None, **read_image_kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        r"""Read the images from file.

        Args:
            root: Optional root directory if the file is a relative path.
            **read_image_kwargs: Optional parameters passed to
                :func:`pystiche.image.read_image`.

        Returns:
            Dictionary with the name image pairs.
        """
        return {
            name: image.read(root=root, **read_image_kwargs) for name, image in self
        }

    def _named_children(self) -> Iterator[Tuple[str, Any]]:
        yield from iter(self._images.items())
