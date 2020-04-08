from pystiche.misc import warn_deprecation

from .transforms.processing import *

# If removed, also remove import in pystiche.image.__init__.py
warn_deprecation(
    "module",
    "pystiche.image.processing",
    "0.4",
    info="It was moved to pystiche.image.transforms.processing",
)
