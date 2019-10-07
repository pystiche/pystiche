from typing import Sequence, Tuple, Dict, Iterator
import torch
from torch import nn
import torch.nn.functional as F
import pystiche
from pystiche.typing import is_conv_module, is_pool_module
from pystiche.misc import verify_str_arg

__all__ = ["Encoder"]


class Encoder(nn.Sequential):
    def forward(
        self, x: torch.Tensor, layers: Sequence[str]
    ) -> Tuple[torch.Tensor, ...]:
        encs_dict = {}
        for name, module in self._necessary_named_modules(layers):
            x = encs_dict[name] = module(x)
        return tuple([encs_dict[name] for name in layers])

    def trim(self, layers: Sequence[str]):
        last_module_name = self._find_last_module_name(layers)
        idx = list(self.children_names()).index(last_module_name)
        del self[idx + 1 :]

    def children_names(self) -> Iterator[str]:
        for name, module in self.named_children():
            yield name

    def _necessary_named_modules(self, layers: Sequence[str]):
        last_module_name = self._find_last_module_name(layers)
        for name, module in self.named_children():
            yield name, module
            if name == last_module_name:
                break

    def _find_last_module_name(self, layers: Sequence[str]) -> str:
        try:
            return sorted(set(layers), key=list(self.children_names()).index)[-1]
        except ValueError as e:
            name = str(e).split()[0]
        raise ValueError("Layer {0} is not part of the encoder".format(name))

    def propagate_guide(
        self, guide: torch.Tensor, layers: Sequence[str], method: str = "simple"
    ) -> Tuple[torch.Tensor, ...]:
        verify_str_arg(method, "method", ("simple", "inside", "all"))
        guides_dct = {}
        for name, module in self._necessary_named_modules(layers):
            if is_pool_module(module):
                guide = F.max_pool2d(guide, **pystiche.pool_module_meta(module))
            # TODO: deal with convolution that doesn't preserve the output shape
            elif is_conv_module(module) and method != "simple":
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
                guide = torch.clamp(guide_folded, 0.0, 1.0)

            guides_dct[name] = guide

        return tuple([guides_dct[name] for name in layers])
