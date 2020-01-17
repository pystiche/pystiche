from typing import Sequence
from copy import copy
import torch
from torch import nn
from pystiche.enc import Encoder

__all__ = ["MultiOperatorEncoder"]


class MultiOperatorEncoder(Encoder):
    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        self._encoder = encoder
        self.layers = set()
        self._encoding_storage = {}

    def register_layer(self, layer: str):
        self.layers.add(layer)

    def reset_layers(self):
        self.layers = set()
        self.reset_storage()

    def reset_storage(self):
        self._encoding_storage = {}

    def encode(self, image: torch.Tensor):
        if not self.layers:
            return

        encs = self._encoder(image, self.layers)
        self._encoding_storage = dict(zip(self.layers, encs))

    def forward(self, image: torch.Tensor, layers: Sequence[str]):
        storage = copy(self._encoding_storage)
        diff_layers = [layer for layer in layers if layer not in storage.keys()]
        if diff_layers:
            encs = self._encoder(image, diff_layers)
            storage.update(dict(zip(diff_layers, encs)))

        return tuple([storage[name] for name in layers])

    def __getattr__(self, name):
        try:
            return getattr(self._encoder, name)
        except AttributeError:
            pass
        msg = f"'{type(self).__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def trim(self):
        self._encoder.trim(self.layers)

    def __contains__(self, name: str) -> bool:
        return name in self._encoder
