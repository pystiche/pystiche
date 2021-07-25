from typing import Any, cast

import PIL.Image

from torch import nn
from torchvision.transforms import Normalize as _Normalize

try:
    from torchvision.transforms.functional import InterpolationMode
except ImportError:

    def InterpolationMode(interpolation_mode: str) -> int:
        return cast(int, getattr(PIL.Image, interpolation_mode.upper()))


if not issubclass(_Normalize, nn.Module):

    class Normalize(_Normalize, nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nn.Module.__init__(self)
            _Normalize.__init__(self, *args, **kwargs)


else:
    Normalize = _Normalize  # type: ignore[misc]


__all__ = ["InterpolationMode", "Normalize"]
