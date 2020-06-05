import os
from os import path
from typing import Dict, Optional

from torch import nn
from torchvision.datasets.folder import is_image_file

from ._core import _Image, _ImageCollection


class LocalImage(_Image):
    def __init__(
        self,
        file: str,
        collect_local_guides: bool = True,
        guides: Optional[_ImageCollection] = None,
        transform: Optional[nn.Module] = None,
        note: Optional[str] = None,
    ):
        if collect_local_guides and guides is None:
            guides = self._collect_guides(file)
        super().__init__(file, guides=guides, transform=transform, note=note)

    @staticmethod
    def _collect_guides(file: str) -> Optional["LocalImageCollection"]:
        dir = path.splitext(file)[0]
        if not path.isdir(dir):
            return None

        image_files = [file for file in os.listdir(dir) if is_image_file(file)]
        if not image_files:
            return None

        guides: Dict[str, "LocalImage"] = {}
        for file in image_files:
            region = path.splitext(path.basename(file))[0]
            guides[region] = LocalImage(
                path.join(dir, file), collect_local_guides=False
            )
        return LocalImageCollection(guides)


class LocalImageCollection(_ImageCollection):
    pass
