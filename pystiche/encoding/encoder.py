from typing import Sequence, Tuple, Iterator
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
        self._assert_contains_layers(layers)
        encs_dict = {}
        for name, module in self._required_named_children(layers):
            x = encs_dict[name] = module(x)
        return tuple([encs_dict[name] for name in layers])

    def trim(self, layers: Sequence[str]):
        self._assert_contains_layers(layers)
        last_module_name = self._find_last_required_children_name(layers)
        idx = list(self.children_names()).index(last_module_name)
        del self[idx + 1 :]

    def propagate_guide(
        self, guide: torch.Tensor, layers: Sequence[str], method: str = "simple"
    ) -> Tuple[torch.Tensor, ...]:
        verify_str_arg(method, "method", ("simple", "inside", "all"))
        guides_dct = {}
        for name, module in self._required_named_children(layers):
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

    def children_names(self) -> Iterator[str]:
        for name, child in self.named_children():
            yield name

    def __contains__(self, name: str) -> bool:
        return name in self.children_names()

    def _assert_contains_layers(self, layers: Sequence[str]):
        for layer in layers:
            if layer not in self:
                raise ValueError(f"Layer {layer} is not part of the encoder.")

    def _find_last_required_children_name(self, layers: Sequence[str]) -> str:
        return sorted(set(layers), key=list(self.children_names()).index)[-1]

    def _required_named_children(self, layers: Sequence[str]):
        last_required_children_name = self._find_last_required_children_name(layers)
        for name, child in self.named_children():
            yield name, child
            if name == last_required_children_name:
                break
