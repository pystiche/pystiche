import warnings

from pystiche.misc import build_deprecation_message

from ._color import *
from ._crop import *
from ._misc import *
from ._motif import *
from ._resize import *

msg = build_deprecation_message(
    "Using functionality from pystiche.image.transforms.functional",
    "0.7.0",
    info="See https://github.com/pmeier/pystiche/issues/382 for details.",
)
warnings.warn(msg, UserWarning)
