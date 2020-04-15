from typing import Any, Dict, Iterator, Optional, Tuple

import pystiche

from .image import DownloadableImage, Image

__all__ = ["ImageCollection", "DownloadableImageCollection"]


class ImageCollection(pystiche.ComplexObject):
    def __init__(
        self, images: Dict[str, Image], root: Optional[str] = None,
    ):
        self.images = images
        # TODO: is root needed here?
        if root is None:
            root = pystiche.home()
        self.root = root

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["root"] = self.root
        dct["num_images"] = len(self)
        return dct

    def _named_children(self) -> Iterator[Tuple[str, Any]]:
        yield from iter(self.images.items())

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item]


class DownloadableImageCollection(ImageCollection):
    def __init__(
        self,
        images: Dict[str, DownloadableImage],
        root: Optional[str] = None,
        download: bool = True,
        overwrite: bool = False,
    ):
        super().__init__(images, root=root)
        if download:
            self.download(overwrite=overwrite)

    def download(self, overwrite: bool = False):
        for image in self.images.values():
            image.download(root=self.root, overwrite=overwrite)
