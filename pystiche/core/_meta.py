from typing import Any, Dict, Sequence, Tuple, Union

import torch

from pystiche.typing import ConvModule, PoolModule

__all__ = [
    "tensor_meta",
    "is_scalar_tensor",
    "conv_module_meta",
    "pool_module_meta",
]


def _extract_meta_attrs(
    obj: Any, attrs: Sequence[str], **kwargs: Any
) -> Dict[str, Any]:
    for attr in attrs:
        if attr not in kwargs:
            kwargs[attr] = getattr(obj, attr)
    return kwargs


TensorMeta = Union[torch.device, torch.dtype]


def tensor_meta(x: torch.Tensor, **kwargs: TensorMeta) -> Dict[str, TensorMeta]:
    attrs = ("dtype", "device")
    return _extract_meta_attrs(x, attrs, **kwargs)


def is_scalar_tensor(x: torch.Tensor) -> bool:
    return x.dim() == 0


ConvModuleMeta = Tuple[int, ...]


def conv_module_meta(
    x: ConvModule, **kwargs: ConvModuleMeta
) -> Dict[str, ConvModuleMeta]:
    attrs = ("kernel_size", "stride", "padding", "dilation")
    return _extract_meta_attrs(x, attrs, **kwargs)


PoolModuleMeta = Tuple[int, ...]


def pool_module_meta(
    x: PoolModule, **kwargs: PoolModuleMeta
) -> Dict[str, PoolModuleMeta]:
    attrs = ("kernel_size", "stride", "padding")
    return _extract_meta_attrs(x, attrs, **kwargs)
