from pystiche.misc import warn_deprecation

from .transforms.processing import *  # noqa: F401, F403

# If removed, also remove import in pystiche.image.__init__.py
warn_deprecation(
    "The module pystiche.image.processing",
    "0.4.0",
    info="It was moved to pystiche.image.transforms.processing",
)
