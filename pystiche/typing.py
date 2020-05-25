import warnings

from pystiche.meta import ConvModule, PoolModule, is_conv_module, is_pool_module
from pystiche.misc import build_deprecation_message

__all__ = [
    "ConvModule",
    "is_conv_module",
    "PoolModule",
    "is_pool_module",
]


msg = build_deprecation_message(
    "The module pystiche.typing",
    "0.4.0",
    info="The functionality was moved to pystiche.meta.",
)
warnings.warn(msg)
