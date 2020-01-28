import torch
from torch import nn
import torch.nn.functional as F
import pystiche
from pystiche.typing import ConvModule, is_conv_module, PoolModule, is_pool_module
from pystiche.misc import verify_str_arg

__all__ = ["propagate_guide"]


def propagate_guide(
    module: nn.Module, guide: torch.Tensor, method: str = "simple"
) -> torch.Tensor:
    verify_str_arg(method, "method", ("simple", "inside", "all"))
    if is_conv_module(module):
        return _conv_guide(module, guide, method)
    elif is_pool_module(module):
        return _pool_guide(module, guide)
    else:
        return guide


def _conv_guide(module: ConvModule, guide: torch.Tensor, method: str) -> torch.Tensor:
    # TODO: deal with convolution that doesn't preserve the output shape
    if method == "simple":
        return guide

    meta = pystiche.conv_module_meta(module)
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
    return F.max_pool2d(guide, **pystiche.pool_module_meta(module))
