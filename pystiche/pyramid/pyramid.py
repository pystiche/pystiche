from typing import Union, Optional, Sequence
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import pystiche
from pystiche.misc import zip_equal
from .level import PyramidLevel


__all__ = ["ImagePyramid", "OctaveImagePyramid"]


class ImagePyramid(pystiche.Object):
    def __init__(
        self,
        edge_sizes: Sequence[int],
        num_steps: Union[Sequence[int], int],
        edges: Union[Sequence[str], str] = "short",
        interpolation_mode: str = "bilinear",
        resize_targets: Optional[Sequence[pystiche.StateObject]] = None,
    ):
        num_levels = len(edge_sizes)
        if isinstance(num_steps, int):
            num_steps = [num_steps] * num_levels
        if isinstance(edges, str):
            edges = [edges] * num_levels

        self._levels = [
            PyramidLevel(
                edge_size, num_steps_, edge, interpolation_mode=interpolation_mode
            )
            for edge_size, num_steps_, edge in zip_equal(edge_sizes, num_steps, edges)
        ]

        self.resize_targets = resize_targets

    def __len__(self):
        return len(self._levels)

    def __getitem__(self, idx):
        return self._levels[idx]

    def __iter__(self):
        state_dicts = self._extract_states()
        for level in self._levels:
            self._resize(level)
            yield level
            self._restore_states(state_dicts)

    def _extract_states(self):
        states = OrderedDict()
        if self.resize_targets is None:
            return states

        for obj in set(self.resize_targets):
            if isinstance(obj, torch.Tensor):
                # FIXME
                state = None
            elif isinstance(obj, (nn.Module, pystiche.StateObject)):
                state = obj.state_dict()
            else:
                state = None

            states[obj] = state
        return states

    def _restore_states(self, states):
        for obj, state in states.items():
            if state is None:
                continue

            if isinstance(obj, torch.Tensor):
                # FIXME
                pass
            else:  # FIXME
                obj.load_state_dict(state)

    def _resize(self, level: PyramidLevel):
        pass


class OctaveImagePyramid(ImagePyramid):
    def __init__(
        self,
        max_edge_size: int,
        num_steps: Union[Sequence[int], int],
        num_levels: Optional[int] = None,
        min_edge_size: int = 64,
        **kwargs
    ):
        if num_levels is None:
            num_levels = int(np.floor(np.log2(max_edge_size / min_edge_size))) + 1

        edge_sizes = [
            round(max_edge_size / (2.0 ** ((num_levels - 1) - level)))
            for level in range(num_levels)
        ]
        super().__init__(edge_sizes, num_steps, **kwargs)
