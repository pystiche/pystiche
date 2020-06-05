import os

from .__about__ import *
from .core import *

os.makedirs(home(), exist_ok=True)  # noqa: F405
