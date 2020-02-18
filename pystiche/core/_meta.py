from typing import Any, Union, Sequence, Dict
import torch
from pystiche.typing import ConvModule, PoolModule


__all__ = [
    "tensor_meta",
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


ConvModuleMeta = Union[int, Sequence[int]]


def conv_module_meta(
    x: ConvModule, **kwargs: ConvModuleMeta
) -> Dict[str, ConvModuleMeta]:
    attrs = ("kernel_size", "stride", "padding", "dilation")
    return _extract_meta_attrs(x, attrs, **kwargs)


PoolModuleMeta = Union[int, Sequence[int]]


def pool_module_meta(
    x: PoolModule, **kwargs: PoolModuleMeta
) -> Dict[str, PoolModuleMeta]:
    attrs = ("kernel_size", "stride", "padding")
    return _extract_meta_attrs(x, attrs, **kwargs)
