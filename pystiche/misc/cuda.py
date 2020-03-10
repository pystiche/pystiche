from typing import Callable
import warnings
from contextlib import contextmanager


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


@contextmanager
def use_cuda_out_of_memory_error():
    def is_cuda_out_of_memory_error(error):
        return str(error).startswith("CUDA out of memory")

    try:
        yield
        return
    except RuntimeError as error:
        if is_cuda_out_of_memory_error(error):
            error = CudaOutOfMemoryError(error)
        raise error


def abort_if_cuda_memory_exausts(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            with use_cuda_out_of_memory_error():
                return fn(*args, **kwargs)
        except CudaOutOfMemoryError as error:
            msg = f"Aborting excecution of {fn.__name__}(). {str(error)}"
            warnings.warn(msg, CudaOutOfMemoryWarning)
            return None

    return wrapper
