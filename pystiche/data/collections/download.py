import os
from os import path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torchvision.datasets.utils import check_md5

import pystiche
from pystiche.misc import download_file

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
        prefix_guide_files: bool = True,
        transform: Optional[nn.Module] = None,
        note: Optional[str] = None,
    ) -> None:
        if file is None:
            file = self.generate_file(url, title, author)

        if guides is not None and prefix_guide_files:
            prefix = path.splitext(path.basename(file))[0]
            for _, guide in guides:
                if not path.isabs(guide.file):
                    guide.file = path.join(prefix, guide.file)

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

        def format(x: str) -> str:
            return "_".join(x.lower().split())

        name_parts = [format(part) for part in (title, author) if part is not None]
        name = "__".join(name_parts)
        ext = path.splitext(url)[1]
        return name + ext

    def download(self, root: Optional[str] = None, overwrite: bool = False) -> None:
        def _download(file: str) -> None:
            os.makedirs(path.dirname(file), exist_ok=True)
            download_file(self.url, file)

        if root is None:
            root = pystiche.home()

        if isinstance(self.guides, DownloadableImageCollection):
            self.guides.download(root=root, overwrite=overwrite)

        file = self.file
        if not path.isabs(file) and root is not None:
            file = path.join(root, file)

        if not path.isfile(file):
            _download(file)
            return

        msg_overwrite = "If you want to overwrite it, set overwrite=True."

        if self.md5 is None:
            if overwrite:
                _download(file)
                return
            else:
                msg = f"{path.basename(file)} already exists in {root}. {msg_overwrite}"
                raise FileExistsError(msg)

        if not check_md5(file, self.md5):
            if overwrite:
                _download(file)
                return
            else:
                msg = (
                    f"{path.basename(file)} with a different MD5 hash already exists "
                    f"in {root}. {msg_overwrite}"
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
    def download(self, root: Optional[str] = None, overwrite: bool = False) -> None:
        for _, image in self:
            if isinstance(image, DownloadableImage):
                image.download(root=root, overwrite=overwrite)

    def read(
        self,
        root: Optional[str] = None,
        download: Optional[bool] = None,
        overwrite: bool = False,
        **read_image_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        return {
            name: image.read(
                root=root, download=download, overwrite=overwrite, **read_image_kwargs
            )
            for name, image in self
        }
