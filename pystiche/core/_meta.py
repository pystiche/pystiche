import warnings
from typing import Any, Callable

from pystiche import meta
from pystiche.misc import build_deprecation_message


def deprecation(fn: Callable) -> Callable:
    name = f"{fn.__name__}()"
    msg = build_deprecation_message(
        f"The function pystiche.{name}",
        "0.4.0",
        info=f"It was moved to pystiche.meta.{name}.",
    )

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(msg)
        return fn(*args, **kwargs)

    return wrapper


__all__ = []
for fn in ("tensor_meta", "is_scalar_tensor", "conv_module_meta", "pool_module_meta"):
    globals()[fn] = deprecation(meta.__dict__[fn])
    __all__.append(fn)
