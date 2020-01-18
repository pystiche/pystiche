from typing import Sequence, Tuple, Iterator, Union, Dict, Optional, Collection
from abc import abstractmethod
from collections import OrderedDict
from copy import copy
import torch
from torch import nn
import torch.nn.functional as F
import pystiche
from pystiche.typing import is_conv_module, is_pool_module
from pystiche.misc import verify_str_arg

__all__ = ["Encoder", "SingleLayerEncoder", "MultiLayerEncoder"]


class Encoder(pystiche.Module):
    @abstractmethod
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        pass


class SingleLayerEncoder(Encoder):
    def __init__(self, multi_layer_encoder: "MultiLayerEncoder", layer : str):
        super().__init__()
        self._multi_layer_encoder = multi_layer_encoder
        self.layer = layer

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self._multi_layer_encoder(input_image, layers=(self.layer,))

    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        pass


class MultiLayerEncoder(pystiche.Module):
    def __init__(self, *args: Union[nn.Module, Dict[str, nn.Module]]) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

        self._registered_layers = set()
        self._storage = {}

        self.requires_grad_(False)
        self.eval()

    def extract_single_layer_encoder(self, layer: str) -> SingleLayerEncoder:
        self._registered_layers.add(layer)
        return SingleLayerEncoder(self, layer)

    def encode(self, image: torch.Tensor):
        if not self._registered_layers:
            return

        encs = self(image, layers=self._registered_layers, store=True)
        self._storage = dict(zip(self._registered_layers, encs))

    def clear_storage(self):
        self._storage = {}

    def forward(
        self, x: torch.Tensor, layers: Sequence[str] = None, store=False
    ) -> Tuple[torch.Tensor, ...]:
        storage = copy(self._storage)
        diff_layers = set(layers) - set(storage.keys())
        for name, module in self._required_named_children(diff_layers):
            x = storage[name] = module(x)

        if store:
            self._storage = storage

        return tuple([storage[name] for name in layers])

    def trim(self, layers: Optional[Collection[str]] = None):
        if layers is None:
            layers = self._registered_layers
        last_module_name = self._find_last_required_children_name(layers)
        idx = list(self.children_names()).index(last_module_name)
        del self[idx + 1 :]

    def children_names(self) -> Iterator[str]:
        for name, child in self.named_children():
            yield name

    def __contains__(self, name: str) -> bool:
        return name in self.children_names()

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


