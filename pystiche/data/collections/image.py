from os import path
from typing import Any, Callable, Dict, Optional

import requests
from PIL import Image as PILImage

import torch
from torchvision.datasets.utils import check_md5

import pystiche
from pystiche.data.license import License, UnknownLicense
from pystiche.image import read_image

__all__ = ["Image", "DownloadableImage"]


class Image(pystiche.ComplexObject):
    def __init__(
        self,
        file: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        date: Optional[str] = None,
        license: Optional[License] = None,
        transform: Optional[Callable[[PILImage.Image], PILImage.Image]] = None,
        note: Optional[str] = None,
        md5: Optional[str] = None,
    ):
        self.file = file
        self.title = title
        self.author = author
        self.date = date

        if license is None:
            license = UnknownLicense()
        self.license = license

        self.transform = transform
        self.note = note
        self.md5 = md5

    def read(
        self, root: Optional[str] = None, **read_image_kwargs: Any,
    ) -> torch.Tensor:
        if root is None:
            root = pystiche.home()
        return read_image(path.join(root, self.file), **read_image_kwargs)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["file"] = self.file
        dct["title"] = self.title if self.title is not None else "unknown"
        dct["author"] = self.author if self.author is not None else "unknown"
        dct["date"] = self.date if self.date is not None else "unknown"
        dct["license"] = self.license
        if self.note is not None:
            dct["note"] = self.note
        return dct


class DownloadableImage(Image):
    def __init__(
        self,
        url: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        file: Optional[str] = None,
        **image_kwargs: Any,
    ):
        if file is None:
            file = self.generate_file(url, title, author)

        super().__init__(file, title=title, author=author, **image_kwargs)
        self.url = url

    @staticmethod
    def generate_file(url: str, title: Optional[str], author: Optional[str]) -> str:
        if title is None and author is None:
            return path.basename(url)

        def format(x):
            return "_".join(x.lower().split())

        name_parts = [format(part) for part in (title, author) if part is not None]
        name = "__".join(name_parts)
        ext = path.splitext(url)[1]
        return name + ext

    def download(self, root: Optional[str] = None, overwrite: bool = False):
        if root is None:
            root = pystiche.home()
        file = path.join(root, self.file)

        def download_and_transform(file: str):
            with open(file, "wb") as fh:
                fh.write(requests.get(self.url).content)

            if self.transform is not None:
                self.transform(PILImage.open(file)).save(file)

        if not path.isfile(file):
            download_and_transform(file)
            return

        msg_tail = "If you want to overwrite it, set overwrite=True."

        if self.md5 is None:
            if overwrite:
                download_and_transform(file)
                return
            else:
                msg = f"{file} already exists in {root}. {msg_tail}"
                raise FileExistsError(msg)

        if not check_md5(file, self.md5):
            if overwrite:
                download_and_transform(file)
                return
            else:
                msg = (
                    f"{file} with a different MD5 hash already exists in {root}. "
                    f"{msg_tail}"
                )
                raise FileExistsError(msg)

    def read(
        self,
        root: Optional[str] = None,
        download: bool = True,
        overwrite: bool = False,
        **read_image_kwargs: Any,
    ) -> torch.Tensor:
        if root is None:
            root = pystiche.home()
        if download:
            self.download(root=root, overwrite=overwrite)
        return super().read(root=root, **read_image_kwargs)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["url"] = self.url
        return dct
