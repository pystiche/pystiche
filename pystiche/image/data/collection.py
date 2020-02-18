from typing import Any, Optional, Tuple, Dict, Callable, Iterator
from os import path
from urllib.request import urlretrieve
from PIL import Image
import torch
import pystiche
from pystiche.image import read_image
from torchvision.datasets.utils import check_md5
from .license import License, UnknownLicense

__all__ = ["DownloadableImage", "DownloadableImageCollection"]


class DownloadableImage(pystiche.Object):
    def __init__(
        self,
        url: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        date: Optional[str] = None,
        license: Optional[License] = None,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        note: Optional[str] = None,
        md5: Optional[str] = None,
        file: Optional[str] = None,
    ):
        self.url = url
        self.title = title
        self.author = author
        self.date = date

        if license is None:
            license = UnknownLicense()
        self.license = license

        self.license = license
        self.transform = transform
        self.note = note
        self.md5 = md5

        if file is None:
            if title is not None or author is not None:
                file = self.generate_file()
            else:
                file = path.basename(url)
        self.file = file

    def generate_file(self) -> str:
        def reformat(x):
            return "_".join(x.lower().split())

        name = f"{reformat(self.title)}__{reformat(self.author)}"
        ext = path.splitext(self.url)[1]
        return name + ext

    def download(self, root: Optional[str] = None, force: bool = False):
        if root is None:
            root = pystiche.home()
        file = path.join(root, self.file)

        def download_and_transform(file: str):
            urlretrieve(self.url, file)

            if self.transform is not None:
                self.transform(Image.open(file)).save(file)

        if not path.isfile(file):
            download_and_transform(file)
            return

        if self.md5 is None or check_md5(file, self.md5):
            return

        if force:
            download_and_transform(file)
            return

        msg = (
            f"{file} with a different MD5 hash is already present in {root}."
            f"If you want to overwrite it, set force=True."
        )
        raise RuntimeError(msg)

    def read(
        self,
        root: Optional[str] = None,
        download: bool = True,
        force: bool = False,
        **read_image_kwargs: Any,
    ) -> torch.Tensor:
        if root is None:
            root = pystiche.home()
        if download:
            self.download(root=root, force=force)
        return read_image(path.join(root, self.file), **read_image_kwargs)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["file"] = self.file
        dct["url"] = self.url
        dct["title"] = self.title if self.title is not None else "unknown"
        dct["author"] = self.author if self.author is not None else "unknown"
        dct["date"] = self.date if self.date is not None else "unknown"
        dct["license"] = self.license
        if self.note is not None:
            dct["note"] = self.note
        return dct


class DownloadableImageCollection(pystiche.Object):
    SEP_CHAR = "-"
    SEP_LINE_LEN = 80

    def __init__(
        self,
        images: Dict[str, DownloadableImage],
        root: Optional[str] = None,
        download: bool = True,
        force: bool = False,
    ):
        self.images = images
        if root is None:
            root = pystiche.home()
        self.root = root

        if download:
            self.download(force=force)

    def download(self, force: bool = False):
        for image in self.images.values():
            image.download(root=self.root, force=force)

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
