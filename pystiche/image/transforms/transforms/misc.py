from typing import Any, Sequence, Dict
import torch
from pystiche.misc import to_engtuplestr
from pystiche.typing import Numeric
from .. import functional as F
from .core import Transform

__all__ = [
    "FloatToUint8Range",
    "Uint8ToFloatRange",
    "ReverseChannelOrder",
    "Normalize",
    "Denormalize",
]


class FloatToUint8Range(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.float_to_uint8_range(x)


class Uint8ToFloatRange(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.uint8_to_float_range(x)


class ReverseChannelOrder(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.reverse_channel_order(x)


class Normalize(Transform):
    def __init__(self, mean: Sequence[Numeric], std: Sequence[Numeric]) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

        self.register_buffer("_mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("_std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, self._mean, self._std)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["mean"] = to_engtuplestr(self.mean)
        dct["std"] = to_engtuplestr(self.std)
        return dct


class Denormalize(Normalize):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.denormalize(x, self._mean, self._std)
