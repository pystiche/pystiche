from typing import Sequence, Tuple, Iterator, Union, Dict, Optional, Collection
from collections import OrderedDict
from copy import copy
import torch
from torch import nn
import pystiche
from .encoder import Encoder
from .guides import propagate_guide

__all__ = ["MultiLayerEncoder", "SingleLayerEncoder"]


class MultiLayerEncoder(pystiche.Module):
    def __init__(self, modules: Dict[str, nn.Module]) -> None:
        super().__init__(named_children=modules)
        self._registered_layers = set()
        self._cache = {}

        self.requires_grad_(False)
        self.eval()

    def children_names(self) -> Iterator[str]:
        for name, child in self.named_children():
            yield name

    def __contains__(self, name: str) -> bool:
        return name in self.children_names()

    def _verify_layer(self, layer: str) -> None:
        if layer not in self:
            raise ValueError(f"Layer {layer} is not part of the encoder.")

    def extract_deepest_layer(self, layers: Collection[str]) -> str:
        for layer in layers:
            self._verify_layer(layer)
        return sorted(set(layers), key=list(self.children_names()).index)[-1]

    def named_children_to(
        self, layer: str, include_last: bool = False
    ) -> Iterator[Tuple[str, pystiche.Module]]:
        idx = list(self.children_names()).index(layer)
        if include_last:
            idx += 1
        return iter(tuple(self.named_children())[:idx])

    def named_children_from(
        self, layer: str, include_first: bool = True
    ) -> Iterator[Tuple[str, pystiche.Module]]:
        idx = list(self.children_names()).index(layer)
        if not include_first:
            idx += 1
        return iter(tuple(self.named_children())[idx:])

    def forward(
        self, x: torch.Tensor, layers: Sequence[str], store=False
    ) -> Tuple[torch.Tensor, ...]:
        storage = copy(self._cache)
        x_key = pystiche.TensorKey(x)
        stored_layers = [
            name for (name, tensor_key) in storage.keys() if tensor_key == x_key
        ]
        diff_layers = set(layers) - set(stored_layers)
        deepest_layer = self.extract_deepest_layer(diff_layers)
        for name, module in self.named_children_to(deepest_layer, include_last=True):
            x = storage[(name, x_key)] = module(x)

        if store:
            self._cache.update(storage)

        return tuple([storage[(name, x_key)] for name in layers])

    def encode(self, image: torch.Tensor):
        if not self._registered_layers:
            return

        image_key = pystiche.TensorKey(image)
        keys = [(layer, image_key) for layer in self._registered_layers]
        encs = self(image, layers=self._registered_layers, store=True)
        self._cache = dict(zip(keys, encs))

    def __getitem__(self, layer: str) -> "SingleLayerEncoder":
        self._verify_layer(layer)
        self._registered_layers.add(layer)
        return SingleLayerEncoder(self, layer)

    def clear_cache(self):
        self._cache = {}

    def trim(self, layers: Optional[Collection[str]] = None):
        if layers is None:
            layers = self._registered_layers
        deepest_layer = self.extract_deepest_layer(layers)
        for name, _ in self.named_children_from(deepest_layer, include_first=False):
            del self._modules[name]

    def propagate_guide(
        self,
        guide: torch.Tensor,
        layers: Sequence[str],
        method: str = "simple",
        allow_empty=False,
    ) -> Tuple[torch.Tensor, ...]:
        guides = {}
        for name, module in self.named_children_to(layers):
            try:
                guide = guides[name] = propagate_guide(
                    module, guide, method=method, allow_empty=allow_empty
                )
            except RuntimeError as error:
                # TODO: customize error message to better reflect which layer causes
                #       the problem
                raise error

        return tuple([guides[name] for name in layers])


class SingleLayerEncoder(Encoder):
    def __init__(self, multi_layer_encoder: MultiLayerEncoder, layer: str):
        super().__init__()
        self.multi_layer_encoder = multi_layer_encoder
        self.layer = layer

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.multi_layer_encoder(input_image, layers=(self.layer,))[0]

    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        return self.multi_layer_encoder.propagate_guide(guide, layers=(self.layer,))[0]

    def __str__(self) -> str:
        name = self.multi_layer_encoder.__class__.__name__
        properties = OrderedDict()
        properties["layer"] = self.layer
        properties.update(self.multi_layer_encoder.properties())
        named_children = ()
        return self._build_str(
            name=name, properties=properties, named_children=named_children
        )
