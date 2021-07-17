try:
    from ._version import version as __version__  # type: ignore[import]
except ImportError:
    __version__ = "UNKNOWN"

from .core import *

from . import loss, enc, demo, data, image, pyramid, optim

import os

os.makedirs(home(), exist_ok=True)
