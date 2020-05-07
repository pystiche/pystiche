import contextlib
import warnings
from typing import Any, Callable, Iterator

__all__ = [
    "CudaOutOfMemoryError",
    "CudaOutOfMemoryWarning",
    "use_cuda_out_of_memory_error",
    "abort_if_cuda_memory_exausts",
]


class CudaOutOfMemoryError(RuntimeError):
    pass


class CudaOutOfMemoryWarning(RuntimeWarning):
    pass


@contextlib.contextmanager
def use_cuda_out_of_memory_error() -> Iterator[None]:
    def is_cuda_out_of_memory_error(error: RuntimeError) -> bool:
        return str(error).startswith("CUDA out of memory")

    try:
        yield
        return
    except RuntimeError as error:
        if is_cuda_out_of_memory_error(error):
            error = CudaOutOfMemoryError(error)
        raise error


def abort_if_cuda_memory_exausts(fn: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            with use_cuda_out_of_memory_error():
                return fn(*args, **kwargs)
        except CudaOutOfMemoryError as error:
            msg = f"Aborting excecution of {fn.__name__}(). {str(error)}"
            warnings.warn(msg, CudaOutOfMemoryWarning)
            return None

    return wrapper
