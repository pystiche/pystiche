import warnings
from collections import OrderedDict, defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import torch
from torch import nn

import pystiche

from ..misc import build_deprecation_message
from .encoder import Encoder
from .guides import propagate_guide

__all__ = ["MultiLayerEncoder", "SingleLayerEncoder"]


class _Layers:
    def __init__(self, modules: Dict[str, nn.Module]) -> None:
        self._modules = modules

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    @property
    def _names(self) -> Tuple[str, ...]:
        # TODO: Check if tuple generation is expensive. If that is the case, cache it
        #  based on self._modules.keys()
        return tuple(self._modules.keys())

    def _name_to_idx(self, name: str) -> int:
        if name not in self:
            raise ValueError

        return self._names.index(name)

    def _idx_to_name(self, idx: int) -> str:
        if not (0 <= idx < len(self)):
            raise ValueError

        return self._names[idx]

    def range(
        self,
        start: Optional[str] = None,
        stop: Optional[str] = None,
        include_start: bool = True,
        include_stop: bool = True,
    ) -> Tuple[str, ...]:
        if not (start or stop):
            return self._names

        if start is None:
            start_idx = 0
        else:
            start_idx = self._name_to_idx(start)
            if not include_start:
                start_idx += 1

        if stop is None:
            stop_idx = len(self)
        else:
            stop_idx = self._name_to_idx(stop)
            if include_stop:
                stop_idx += 1

        return self._names[start_idx:stop_idx]

    def _depth(
        self, names: Iterable[str], extractor: Callable[[List[int]], int]
    ) -> str:
        return self._idx_to_name(extractor([self._name_to_idx(name) for name in names]))

    def shallowest(self, names: Optional[Iterable[str]] = None) -> str:
        return self._depth(names or self._names, min)

    def deepest(self, names: Optional[Iterable[str]] = None) -> str:
        return self._depth(names or self._names, max)

    def _neighbour(
        self,
        name: str,
        names: Iterable[str],
        edge_idx: int,
        extractor: Callable[[int, List[int]], Optional[str]],
    ) -> Optional[str]:
        if not names:
            return None

        idx = self._name_to_idx(name)
        idcs = [self._name_to_idx(name) for name in names]

        if edge_idx in idcs:
            return self._idx_to_name(edge_idx)

        return extractor(idx, idcs)

    def _extract_prev(self, idx: int, idcs: List[int]) -> Optional[str]:
        candidates = [other_idx for other_idx in idcs if other_idx < idx]
        if not candidates:
            return None

        return self._idx_to_name(max(candidates))

    def prev(self, name: str, names: Iterable[str]) -> Optional[str]:
        return self._neighbour(name, names, edge_idx=0, extractor=self._extract_prev)

    def _extract_next(self, idx: int, idcs: List[int]) -> Optional[str]:
        candidates = [other_idx for other_idx in idcs if other_idx > idx]
        if not candidates:
            return None

        return self._idx_to_name(min(candidates))

    def next(self, name: str, names: Collection[str]) -> Optional[str]:
        return self._neighbour(
            name, names, edge_idx=len(self) - 1, extractor=self._extract_next
        )


class MultiLayerEncoder(pystiche.Module):
    r"""Sequential encoder with convenient access to intermediate layers.

    Args:
        modules: Named modules that serve as basis for the encoding.

    Attributes:
        registered_layers: Layers, on which the encodings will be cached during the
            :meth:`forward` pass.
    """

    def __init__(self, modules: Sequence[Tuple[str, nn.Module]]) -> None:
        super().__init__(named_children=modules)
        self._layers: _Layers = _Layers(self._modules)
        self.registered_layers: Set[str] = set()
        self._cache: DefaultDict[torch.Tensor, Dict[str, torch.Tensor]] = defaultdict(
            lambda: {}
        )

    def __contains__(self, layer: str) -> bool:
        r"""Is the layer part of the multi-layer encoder?

        Args:
            layer: Layer to be checked.
        """
        return layer in self._layers

    def _verify(self, name: str) -> None:
        if name not in self:
            raise ValueError(f"Layer {name} is not part of the multi-layer encoder.")

    def register_layer(self, layer: str) -> None:
        r"""Register a layer for caching the encodings in the :meth:`forward` pass.

        Args:
            layer: Layer to be registered.
        """
        self._verify(layer)
        self.registered_layers.add(layer)

    # FIXME: could this be moved into pystiche.Module?
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Invokes :meth:`forward`."""
        return super().__call__(*args, **kwargs)

    def forward(
        self,
        input: torch.Tensor,
        layer: Optional[str] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        to_cache: Optional[Collection[str]] = None,
    ) -> torch.Tensor:
        r"""Encode the input.

        Args:
            input: Input to be encoded.
            layer: Layer on which the ``input`` should be encoded. If omitted, defaults
                to the last layer in the multi-layer encoder.
            cache: Encoding cache. If omitted, defaults to the the internal cache.
            to_cache: Layers, of which the encodings should be cached. If omitted,
                defaults to :attr:`registered_layers`.

        Examples:

            >>> modules = [("conv", nn.Conv2d(3, 3, 3)), ("pool", nn.MaxPool2d(2))]
            >>> mle = pystiche.enc.MultiLayerEncoder(modules)
            >>> input = torch.rand(1, 3, 128, 128)
            >>> output = mle(input, "conv")
        """
        if layer is None:
            layer = tuple(self._modules.keys())[-1]
        else:
            self._verify(layer)

        if cache is None:
            cache = self._cache[input]
            if input.requires_grad:
                input.register_hook(lambda grad: self.clear_cache())
        if layer in cache:
            return cache[layer]

        if to_cache is None:
            to_cache = self.registered_layers

        prev = self._layers.prev(layer, cache.keys())
        if prev is not None:
            input = cache[prev]

        for name in self._layers.range(prev, layer, include_start=False):
            module = self._modules[name]
            input = module(input)

            if name in to_cache:
                cache[name] = input

        return input

    def clear_cache(self) -> None:
        r"""Clear the internal cache."""
        self._cache.clear()

    def empty_storage(self) -> None:
        msg = build_deprecation_message(
            "The method 'empty_storage'", "1.0", info="It was renamed to 'clear_cache'."
        )
        warnings.warn(msg)
        self.clear_cache()

    def encode(
        self, input: torch.Tensor, layers: Sequence[str],
    ) -> Tuple[torch.Tensor, ...]:
        r"""Encode the input on layers.

        Args:
            input: Input to be encoded.
            layers: Layers on which the ``input`` should be encoded.
        """
        cache: Dict[str, torch.Tensor] = {}
        return tuple(
            self(input, layer, cache=cache, to_cache=layers) for layer in layers
        )

    def propagate_guide(
        self,
        guide: torch.Tensor,
        layers: Sequence[str],
        method: str = "simple",
        allow_empty: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        r"""Propagate the guide on the given layers.

        Args:
            guide: Guide.
            layers: Layers.
            allow_empty: If ``True``, allow the propagated guides to become empty.
                Defaults to ``False``.

        Returns:
            Tuple of guides which order corresponds to ``layers``.
        """
        guides = {}
        for name in self._layers.range(stop=self._layers.deepest(layers)):
            module = self._modules[name]
            try:
                guide = guides[name] = propagate_guide(
                    module, guide, method=method, allow_empty=allow_empty
                )
            except RuntimeError as error:
                # TODO: customize error message to better reflect which layer causes
                #       the problem
                raise error

        return tuple(guides[name] for name in layers)

    def trim(self, layers: Optional[Iterable[str]] = None) -> None:
        if layers is None:
            layers = self.registered_layers
        else:
            for name in layers:
                self._verify(name)

        for name in self._layers.range(
            self._layers.deepest(layers), include_start=False
        ):
            del self._modules[name]

    def extract_encoder(self, layer: str) -> "SingleLayerEncoder":
        r"""Extract a :class:`SingleLayerEncoder` for the layer and register it.

        Args:
            layer: Layer.
        """
        self.register_layer(layer)
        return SingleLayerEncoder(self, layer)


class SingleLayerEncoder(Encoder):
    r"""Encoder extracted from a :class:`MultiLayerEncoder` that operates on a single
    layer. Invokes :meth:`SingleLayerEncoder.forward` if called.

    Attributes:
        multi_layer_encoder: Corresponding multi-layer encoder.
        layer: Encoding layer.
    """

    def __init__(self, multi_layer_encoder: MultiLayerEncoder, layer: str):
        super().__init__()
        self.multi_layer_encoder = multi_layer_encoder
        self.layer = layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Encode the given input on :attr:`SingleLayerEncoder.layer` of
        :attr:`SingleLayerEncoder.multi_layer_encoder`.

        Args:
            input_image: Input image.
        """
        return cast(torch.Tensor, self.multi_layer_encoder(input, self.layer))

    def propagate_guide(self, guide: torch.Tensor) -> torch.Tensor:
        r"""Propagate the given guide on :attr:`SingleLayerEncoder.layer` of
        :attr:`SingleLayerEncoder.multi_layer_encoder`.

        Args:
            guide: Guide.
        """
        return self.multi_layer_encoder.propagate_guide(guide, layers=(self.layer,))[0]

    def __repr__(self) -> str:
        name = self.multi_layer_encoder.__class__.__name__
        properties = OrderedDict()
        properties["layer"] = self.layer
        properties.update(self.multi_layer_encoder.properties())
        named_children = ()
        return self._build_repr(
            name=name, properties=properties, named_children=named_children
        )
