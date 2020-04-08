from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image

import torch

from pystiche.image import io

from .core import Transform

__all__ = ["ImportFromPIL", "ExportToPIL"]


class ImportFromPIL(Transform):
    def __init__(
        self, device: Union[torch.device, str] = "cpu", make_batched: bool = True
    ):
        super().__init__()
        self.device = device
        self.make_batched = make_batched

    def forward(self, x: Image.Image) -> torch.Tensor:
        return io.import_from_pil(x, self.device, make_batched=self.make_batched)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        device = str(self.device)
        if device != "cpu":
            dct["device"] = device
        dct["make_batched"] = self.make_batched
        return dct


class ExportToPIL(Transform):
    def __init__(self, mode: Optional[str] = None):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> Union[Image.Image, Tuple[Image.Image, ...]]:
        return io.export_to_pil(x, mode=self.mode)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        if self.mode is not None:
            dct["mode"] = self.mode
        return dct
