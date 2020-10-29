import warnings
from collections import OrderedDict
from copy import copy
from typing import Collection, Dict, Iterator, Optional, Sequence, Set, Tuple, cast

import torch
from torch import nn

import pystiche
from pystiche.misc import suppress_warnings

from .encoder import Encoder
from .guides import propagate_guide

__all__ = ["MultiLayerEncoder", "SingleLayerEncoder"]


def _future_warning(name: str) -> None:
    msg = (
        f"The functionality of MultiLayerEncoder.{name} will change in the future. "
        f"If you depend on this functionality, "
        f"see https://github.com/pmeier/pystiche/issues/435 for details "
    )
    warnings.warn(msg, FutureWarning)


class MultiLayerEncoder(pystiche.Module):
    r"""Sequential architecture that supports extracting encodings from intermediate
    layers in a single forward pass. Encodings can be stored to avoid recalculating
    them if they are required multiple times. Invokes :meth:`MultiLayerEncoder.forward`
    if called.

    Args:
        modules: Named sequential modules.

    Attributes:
        registered_layers: Set of registered layers for
            :meth:`MultiLayerEncoder.encode` and :meth:`MultiLayerEncoder.trim`.
    """

    def __init__(self, modules: Sequence[Tuple[str, nn.Module]]) -> None:
        super().__init__(named_children=modules)
        self.registered_layers: Set[str] = set()
        self._storage: Dict[Tuple[str, pystiche.TensorKey], torch.Tensor] = dict()

        # TODO: remove this?
        self.requires_grad_(False)
        self.eval()

    def children_names(self) -> Iterator[str]:
        for name, child in self.named_children():
            yield name

    def __contains__(self, layer: str) -> bool:
        r"""Checks if the given layer is part of the :class:`MultiLayerEncoder`

        Args:
            layer: Layer.
        """
        return layer in self.children_names()

    def _verify_layer(self, layer: str) -> None:
        if layer not in self:
            raise ValueError(f"Layer {layer} is not part of the encoder.")

    def extract_deepest_layer(self, layers: Collection[str]) -> str:
        for layer in layers:
            self._verify_layer(layer)
        return sorted(set(layers), key=list(self.children_names()).index)[-1]

    def named_children_to(
        self, layer: str, include_last: bool = False
    ) -> Iterator[Tuple[str, nn.Module]]:
        self._verify_layer(layer)
        idx = list(self.children_names()).index(layer)
        if include_last:
            idx += 1
        for name, child in tuple(self.named_children())[:idx]:
            yield name, child

    def named_children_from(
        self, layer: str, include_first: bool = True
    ) -> Iterator[Tuple[str, nn.Module]]:
        self._verify_layer(layer)
        idx = list(self.children_names()).index(layer)
        if not include_first:
            idx += 1
        for name, child in tuple(self.named_children())[idx:]:
            yield name, child

    def forward(
        self, input: torch.Tensor, layers: Sequence[str], store: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        r"""Encode the input on the given layers in a single forward pass. If the input
        was encoded before the encodings are extracted from the storage rather than
        executing the forward pass again.

        Args:
            input: Input.
            layers: Layers.
            store: If ``True``, store the encodings.

        Returns:
            Tuple of encodings which order corresponds to ``layers``.
        """
        _future_warning("__call__")
        storage = copy(self._storage)
        input_key = pystiche.TensorKey(input)
        stored_layers = [name for name, key in storage.keys() if key == input_key]
        diff_layers = set(layers) - set(stored_layers)

        if diff_layers:
            deepest_layer = self.extract_deepest_layer(diff_layers)
            for name, module in self.named_children_to(
                deepest_layer, include_last=True
            ):
                input = storage[(name, input_key)] = module(input)

            if store:
                self._storage = storage

        return tuple(storage[(name, input_key)] for name in layers)

    def extract_encoder(self, layer: str) -> "SingleLayerEncoder":
        r"""Extract a :class:`SingleLayerEncoder` for the given layer and register
        the layer in :attr:`MultiLayerEncoder.registered_layers`.

        Args:
            layer: Layer.
        """
        self._verify_layer(layer)
        self.registered_layers.add(layer)
        return SingleLayerEncoder(self, layer)

    def encode(self, input: torch.Tensor) -> None:
        r"""Encode the given input and store the encodings of all
        :attr:`MultiLayerEncoder.registered_layers`.

        Args:
            input: Input.
        """
        _future_warning("encode")

        if not self.registered_layers:
            return

        key = pystiche.TensorKey(input)
        keys = [(layer, key) for layer in self.registered_layers]
        encs = self(input, layers=self.registered_layers, store=True)
        self._storage = dict(zip(keys, encs))

    def empty_storage(self) -> None:
        r"""Empty the encodings storage."""
        self._storage = {}

    def trim(self, layers: Optional[Collection[str]] = None) -> None:
        r"""Remove excess layers that are not necessary to generate the encodings of
        ``layers``.

        Args:
            layers: Layers that the :class:`MultiLayerEncoder` need to be able to
                generate encodings for. If ``None``,
                :attr:`MultiLayerEncoder.registered_layers` is used. Defaults to
                ``None``.
        """
        if layers is None:
            layers = self.registered_layers
        deepest_layer = self.extract_deepest_layer(layers)
        for name, _ in self.named_children_from(deepest_layer, include_first=False):
            del self._modules[name]

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
        deepest_layer = self.extract_deepest_layer(layers)
        for name, module in self.named_children_to(deepest_layer, include_last=True):
            try:
                guide = guides[name] = propagate_guide(
                    module, guide, method=method, allow_empty=allow_empty
                )
            except RuntimeError as error:
                # TODO: customize error message to better reflect which layer causes
                #       the problem
                raise error

        return tuple(guides[name] for name in layers)


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

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        r"""Encode the given input image on :attr:`SingleLayerEncoder.layer` of
        :attr:`SingleLayerEncoder.multi_layer_encoder`.

        Args:
            input_image: Input image.
        """
        with suppress_warnings(FutureWarning):
            return cast(
                Tuple[torch.Tensor],
                self.multi_layer_encoder(input_image, layers=(self.layer,)),
            )[0]

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
