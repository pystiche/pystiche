import os
from os import path

from .__about__ import *

try:
    here = path.dirname(__file__)
    with open(path.join(here, "__version__"), "r") as fh:
        __version__ = fh.read().strip()
except FileNotFoundError:
    __version__ = __base_version__  # noqa: F405

from .core import *


os.makedirs(home(), exist_ok=True)  # noqa: F405
