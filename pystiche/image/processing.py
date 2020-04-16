import warnings

from pystiche.misc import build_deprecation_message

from .transforms.processing import *  # noqa: F401, F403

# If removed, also remove import in pystiche.image.__init__.py
msg = build_deprecation_message(
    "The module pystiche.image.processing",
    "0.4.0",
    info="It was moved to pystiche.image.transforms.processing",
)
warnings.warn(msg)
