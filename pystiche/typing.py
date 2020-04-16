from pystiche.meta import (
    ConvModule,
    ConvModuleMeta,
    PoolModule,
    PoolModuleMeta,
    TensorMeta,
    is_conv_module,
    is_pool_module,
)
from pystiche.misc import warn_deprecation

__all__ = [
    "TensorMeta",
    "ConvModule",
    "is_conv_module",
    "ConvModuleMeta",
    "PoolModule",
    "is_pool_module",
    "PoolModuleMeta",
]

warn_deprecation(
    "module",
    "pystiche.typing",
    "0.4.0",
    info="The functionality was moved to pystiche.meta",
)
