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


def tensor_meta(x: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
    attrs = ("dtype", "device")
    return _extract_meta_attrs(x, attrs, **kwargs)


def conv_module_meta(x: ConvModule, **kwargs: Any) -> Dict[str, Any]:
    attrs = ("kernel_size", "stride", "padding", "dilation")
    return _extract_meta_attrs(x, attrs, **kwargs)


def pool_module_meta(x: PoolModule, **kwargs: Any) -> Dict[str, Any]:
    attrs = ("kernel_size", "stride", "padding")
    return _extract_meta_attrs(x, attrs, **kwargs)
