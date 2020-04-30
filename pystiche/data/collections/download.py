import os
from os import path
from typing import Any, Dict, Optional

import requests

import torch
from torch import nn
from torchvision.datasets.utils import check_md5

import pystiche

from ..license import License, UnknownLicense
from ._core import _Image, _ImageCollection


class DownloadableImage(_Image):
    def __init__(
        self,
        url: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        date: Optional[str] = None,
        license: Optional[License] = None,
        md5: Optional[str] = None,
        file: Optional[str] = None,
        guides: Optional["DownloadableImageCollection"] = None,
        transform: Optional[nn.Module] = None,
        note: Optional[str] = None,
    ):
        if file is None:
            file = self.generate_file(url, title, author)

        super().__init__(file, guides=guides, transform=transform, note=note)
        self.url = url
        self.title = title
        self.author = author
        self.date = date

        if license is None:
            license = UnknownLicense()
        self.license = license

        self.md5 = md5

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
        def _download(file: str):
            with open(file, "wb") as fh:
                fh.write(requests.get(self.url).content)

        if root is None:
            root = pystiche.home()

        file = self.file
        if not path.isabs(file) and root is not None:
            if root is None:
                root = pystiche.home()
            file = path.join(root, file)

        if self.guides is not None:
            dir = path.splitext(file)[0]
            os.makedirs(dir, exist_ok=True)
            for region, guide in self.guides:
                if isinstance(guide, DownloadableImage):
                    guide.download(root=dir, overwrite=overwrite)

        if not path.isfile(file):
            _download(file)
            return

        msg_overwrite = "If you want to overwrite it, set overwrite=True."

        if self.md5 is None:
            if overwrite:
                _download(file)
                return
            else:
                msg = f"{file} already exists in {root}. {msg_overwrite}"
                raise FileExistsError(msg)

        if not check_md5(file, self.md5):
            if overwrite:
                _download(file)
                return
            else:
                msg = (
                    f"{file} with a different MD5 hash already exists in {root}. "
                    f"{msg_overwrite}"
                )
                raise FileExistsError(msg)

    def read(
        self,
        root: Optional[str] = None,
        download: Optional[bool] = None,
        overwrite: bool = False,
        **read_image_kwargs: Any,
    ) -> torch.Tensor:
        if root is None:
            root = pystiche.home()
        if download is None:
            file_exists = path.isfile(path.join(root, self.file))
            md5_available = self.md5 is not None
            download = False if file_exists and not md5_available else True
        if download:
            self.download(root=root, overwrite=overwrite)
        return super().read(root=root, **read_image_kwargs)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["url"] = self.url
        dct["title"] = self.title if self.title is not None else "unknown"
        dct["author"] = self.author if self.author is not None else "unknown"
        dct["date"] = self.date if self.date is not None else "unknown"
        dct["license"] = self.license
        return dct


class DownloadableImageCollection(_ImageCollection):
    def download(self, root: Optional[str] = None, overwrite: bool = False):
        for _, image in self:
            image.download(root=root, overwrite=overwrite)

    def read(
        self,
        root: Optional[str] = None,
        download: Optional[bool] = None,
        overwrite: bool = False,
        **read_image_kwargs: Any,
    ):
        return {
            name: image.read(
                root=root, download=download, overwrite=overwrite, **read_image_kwargs
            )
            for name, image in self
        }
