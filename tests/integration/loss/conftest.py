import pytest

import torch
from torch import nn

from pystiche import enc


class NoOpEncoder(enc.Encoder):
    def forward(self, input):
        return input

    def propagate_guide(self, guide):
        return guide


@pytest.fixture
def noop_encoder():
    return NoOpEncoder()


@pytest.fixture
def encoder():
    return enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))
