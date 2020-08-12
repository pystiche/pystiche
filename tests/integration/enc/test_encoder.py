import pytorch_testing_utils as ptu

import torch
from torch import nn

from pystiche import enc


def test_SequentialEncoder_call():
    torch.manual_seed(0)
    modules = (nn.Conv2d(3, 3, 3), nn.ReLU())
    input = torch.rand(1, 3, 256, 256)

    pystiche_encoder = enc.SequentialEncoder(modules)
    torch_encoder = nn.Sequential(*modules)

    actual = pystiche_encoder(input)
    desired = torch_encoder(input)
    ptu.assert_allclose(actual, desired)
