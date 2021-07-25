# type: ignore

import functools
import warnings
from typing import Callable

import pystiche.loss.functional as F
from pystiche.misc import build_deprecation_message


def _deprecate(fn: Callable) -> Callable:
    name = fn.__name__
    msg = build_deprecation_message(
        f"The function ops.functional.{name}",
        "1.0",
        info=(
            f"It was moved to loss.functional.{name}. "
            f"See https://github.com/pystiche/pystiche/issues/436 for details"
        ),
    )

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        warnings.warn(msg)
        return fn(*args, **kwargs)

    return wrapper


__all__ = []
for name in dir(F):
    if name.startswith("_") or "loss" not in name:
        continue

    fn = getattr(F, name)
    if not callable(fn):
        continue

    __all__.append(name)
    globals()[name] = _deprecate(fn)
