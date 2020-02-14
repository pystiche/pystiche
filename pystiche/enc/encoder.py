from typing import Sequence, Tuple, Iterator, Union, Dict, Optional, Collection
from abc import abstractmethod
from collections import OrderedDict
from copy import copy
import torch
from torch import nn
import pystiche
from .guides import propagate_guide

__all__ = ["Encoder", "SequentialEncoder", "SingleLayerEncoder", "MultiLayerEncoder"]


class Encoder(pystiche.Module):
    @abstractmethod
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        pass


class SequentialEncoder(Encoder):
    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__()
        self.add_indexed_modules(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x

    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            guide = propagate_guide(module, guide)
        return guide


class SingleLayerEncoder(Encoder):
    def __init__(self, multi_layer_encoder: "MultiLayerEncoder", layer: str):
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


class MultiLayerEncoder(pystiche.Module):
    def __init__(self, modules: Dict[str, nn.Module]) -> None:
        super().__init__()
        self.add_named_modules(modules)
        self._registered_layers = set()
        self._storage = {}

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

    def shallowest_layer(self, layers: Collection[str]) -> str:
        for layer in layers:
            self._verify_layer(layer)
        return sorted(set(layers), key=list(self.children_names()).index)[0]

    def deepest_layer(self, layers: Collection[str]) -> str:
        for layer in layers:
            self._verify_layer(layer)
        return sorted(set(layers), key=list(self.children_names()).index)[-1]

    def named_children_up_to(
        self, layer: Union[str, Collection[str]], include_last: bool = True
    ) -> Iterator[Tuple[str, pystiche.Module]]:
        if not layer:
            return iter(())
        elif not isinstance(layer, str):
            layer = self.deepest_layer(layer)
        idx = list(self.children_names()).index(layer)
        if include_last:
            idx += 1
        return iter(list(self.named_children())[:idx])

    def named_children_from(
        self, layer: Union[str, Collection[str]], include_first: bool = True
    ) -> Iterator[Tuple[str, pystiche.Module]]:
        if not layer:
            return iter(())
        elif not isinstance(layer, str):
            layer = self.deepest_layer(layer)
        idx = list(self.children_names()).index(layer)
        if not include_first:
            idx += 1
        return iter(list(self.named_children())[idx:])

    def __getitem__(self, layer: str) -> SingleLayerEncoder:
        self._verify_layer(layer)
        self._registered_layers.add(layer)
        return SingleLayerEncoder(self, layer)

    def __delitem__(self, layer: str):
        self._verify_layer(layer)
        del self._modules[layer]

    def forward(
        self, x: torch.Tensor, layers: Sequence[str], store=False
    ) -> Tuple[torch.Tensor, ...]:
        storage = copy(self._storage)
        diff_layers = set(layers) - set(storage.keys())
        for name, module in self.named_children_up_to(diff_layers, include_last=True):
            x = storage[name] = module(x)

        if store:
            self._storage.update(storage)

        return tuple([storage[name] for name in layers])

    def encode(self, image: torch.Tensor):
        if not self._registered_layers:
            return

        encs = self(image, layers=self._registered_layers, store=True)
        self._storage = dict(zip(self._registered_layers, encs))

    def clear_storage(self):
        self._storage = {}

    def trim(self, layers: Optional[Collection[str]] = None):
        if layers is None:
            layers = self._registered_layers
        for name, _ in self.named_children_from(layers, include_first=False):
            del self[name]

    def propagate_guide(
        self,
        guide: torch.Tensor,
        layers: Sequence[str],
        method: str = "simple",
        allow_empty=False,
    ) -> Tuple[torch.Tensor, ...]:
        guides = {}
        for name, module in self.named_children_up_to(layers):
            try:
                guide = guides[name] = propagate_guide(
                    module, guide, method=method, allow_empty=allow_empty
                )
            except RuntimeError as error:
                # TODO: customize error message to better reflect which layer causes
                #       the problem
                raise error

        return tuple([guides[name] for name in layers])
