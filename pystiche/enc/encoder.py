from typing import Sequence, Tuple, Iterator, Union, Dict, Optional, Collection
from copy import copy
import torch
from torch import nn
import torch.nn.functional as F
import pystiche
from pystiche.typing import is_conv_module, is_pool_module
from pystiche.misc import verify_str_arg

__all__ = ["Encoder"]


class Encoder(nn.Sequential):
    def __init__(self, *args: Union[nn.Module, Dict[str, nn.Module]]) -> None:
        super().__init__(*args)
        self.layers = set()
        self._storage = {}

        self.requires_grad_(False)
        self.eval()

    def register_layer(self, layer: str):
        self.layers.add(layer)

    def __contains__(self, name: str) -> bool:
        return name in self.children_names()

    def clear_storage(self):
        self._storage = {}

    def encode(self, image: torch.Tensor):
        # # this is only here to not run into the backward second time error
        # self._storage = {}

        if not self.layers:
            return

        encs = self(image, layers=self.layers, store=True)
        self._storage = dict(zip(self.layers, encs))

    def forward(
        self, x: torch.Tensor, layers: Optional[Sequence[str]] = None, store=False
    ) -> Tuple[torch.Tensor, ...]:
        if layers is None:
            layers = self.layers

        storage = copy(self._storage)
        diff_layers = set(layers) - set(storage.keys())
        for name, module in self._required_named_children(diff_layers):
            x = storage[name] = module(x)

        if store:
            self._storage = storage

        return tuple([storage[name] for name in layers])

    def trim(self, layers: Optional[Collection[str]] = None):
        if layers is None:
            layers = self.layers
        last_module_name = self._find_last_required_children_name(layers)
        idx = list(self.children_names()).index(last_module_name)
        del self[idx + 1 :]

    def children_names(self) -> Iterator[str]:
        for name, child in self.named_children():
            yield name

    def _verify_layer(self, layer: str) -> None:
        if layer not in self:
            raise ValueError(f"Layer {layer} is not part of the encoder.")

    def _find_last_required_children_name(self, layers: Collection[str]) -> str:
        for layer in layers:
            self._verify_layer(layer)
        return sorted(set(layers), key=list(self.children_names()).index)[-1]

    def _required_named_children(self, layers: Collection[str]):
        if not layers:
            return
            # yield?

        last_required_children_name = self._find_last_required_children_name(layers)
        for name, child in self.named_children():
            yield name, child
            if name == last_required_children_name:
                break

    # def propagate_guide(
    #     self, guide: torch.Tensor, layers: Sequence[str], method: str = "simple"
    # ) -> Tuple[torch.Tensor, ...]:
    #     verify_str_arg(method, "method", ("simple", "inside", "all"))
    #     guides_dct = {}
    #     for name, module in self._required_named_children(layers):
    #         if is_pool_module(module):
    #             guide = F.max_pool2d(guide, **pystiche.pool_module_meta(module))
    #         # TODO: deal with convolution that doesn't preserve the output shape
    #         elif is_conv_module(module) and method != "simple":
    #             meta = pystiche.conv_module_meta(module)
    #             guide_unfolded = F.unfold(guide, **meta).byte()
    #
    #             if method == "inside":
    #                 mask = ~torch.all(guide_unfolded, 1, keepdim=True)
    #                 val = False
    #             else:
    #                 mask = torch.any(guide_unfolded, 1, keepdim=True)
    #                 val = True
    #
    #             mask, _ = torch.broadcast_tensors(mask, guide_unfolded)
    #             guide_unfolded[mask] = val
    #
    #             guide_folded = F.fold(guide_unfolded.float(), guide.size()[2:], **meta)
    #             guide = torch.clamp(guide_folded, 0.0, 1.0)
    #
    #         guides_dct[name] = guide
    #
    #     return tuple([guides_dct[name] for name in layers])
