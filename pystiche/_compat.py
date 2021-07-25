from typing import cast

import PIL.Image

try:
    from torchvision.transforms.functional import InterpolationMode
except ImportError:

    def InterpolationMode(interpolation_mode: str) -> int:
        return cast(int, getattr(PIL.Image, interpolation_mode.upper()))


__all__ = ["InterpolationMode"]
