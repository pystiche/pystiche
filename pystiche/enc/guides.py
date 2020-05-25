from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from pystiche.meta import (
    ConvModule,
    PoolModule,
    conv_module_meta,
    is_conv_module,
    is_pool_module,
    pool_module_meta,
)
from pystiche.misc import verify_str_arg

__all__ = ["propagate_guide"]


def propagate_guide(
    module: nn.Module,
    guide: torch.Tensor,
    method: str = "simple",
    allow_empty: bool = False,
) -> torch.Tensor:
    verify_str_arg(method, "method", ("simple", "inside", "all"))
    if is_conv_module(module):
        guide = _conv_guide(cast(ConvModule, module), guide, method)
    elif is_pool_module(module):
        guide = _pool_guide(cast(PoolModule, module), guide)

    if allow_empty or torch.any(guide.bool()):
        return guide

    msg = (
        f"Guide has no longer any entries after propagation through "
        f"{module.__class__.__name__}({module.extra_repr()}). If this is valid, "
        f"set allow_empty=True."
    )
    raise RuntimeError(msg)


def _conv_guide(module: ConvModule, guide: torch.Tensor, method: str) -> torch.Tensor:
    # TODO: deal with convolution that doesn't preserve the output shape
    if method == "simple":
        return guide

    meta = conv_module_meta(module)
    guide_unfolded = F.unfold(guide, **meta).byte()

    if method == "inside":
        mask = ~torch.all(guide_unfolded, 1, keepdim=True)
        val = False
    else:
        mask = torch.any(guide_unfolded, 1, keepdim=True)
        val = True

    mask, _ = torch.broadcast_tensors(mask, guide_unfolded)
    guide_unfolded[mask] = val

    guide_folded = F.fold(guide_unfolded.float(), guide.size()[2:], **meta)
    return torch.clamp(guide_folded, 0.0, 1.0)


def _pool_guide(module: PoolModule, guide: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, F.max_pool2d(guide, **pool_module_meta(module)))
