from typing import Sequence
from copy import copy
import torch
import pystiche
from pystiche.enc import Encoder

__all__ = ["MultiOperatorEncoder"]


class MultiOperatorEncoder(pystiche.object):
    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        self._encoder = encoder
        self.layers = None
        self._encoding_storage = None
        self.reset_layers()

    def register_layers(self, layers: Sequence[str]):
        self.layers.update(layers)

    def reset_layers(self):
        self.layers = set()
        self.clear_storage()

    def encode(self, image: torch.Tensor):
        if not self.layers:
            return

        encs = self._encoder(image, self.layers)
        self._encoding_storage = dict(zip(self.layers, encs))

    def clear_storage(self):
        self._encoding_storage = {}

    def __call__(self, image: torch.Tensor, layers: Sequence[str]):
        storage = copy(self._encoding_storage)
        diff_layers = [layer for layer in layers if layer not in storage.keys()]
        if diff_layers:
            encs = self._encoder(image, diff_layers)
            storage.update(dict(zip(diff_layers, encs)))

        return pystiche.tuple([storage[name] for name in layers])

    def verify_layers(self, layers: Sequence[str]):
        return self._encoder.verify_layers(layers)

    def __contains__(self, name: str) -> bool:
        return self._encoder.__contains__(name)

    def trim(self):
        self._encoder.trim(self.layers)

    def propagate_guide(self, *args, **kwargs):
        return pystiche.tuple(self._encoder.propagate_guide(*args, **kwargs))
