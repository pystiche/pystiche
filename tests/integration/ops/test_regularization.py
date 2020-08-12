import pytorch_testing_utils as ptu

import torch

import pystiche.ops.functional as F
from pystiche import ops


def test_TotalVariationOperator_call():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 128, 128)
    exponent = 3.0

    op = ops.TotalVariationOperator(exponent=exponent)

    actual = op(image)
    desired = F.total_variation_loss(image, exponent=exponent)
    ptu.assert_allclose(actual, desired)


def test_ValueRangeOperator_call():
    torch.manual_seed(0)
    image = torch.randn(1, 3, 128, 128)

    op = ops.ValueRangeOperator()

    actual = op(image)
    desired = F.value_range_loss(image)
    ptu.assert_allclose(actual, desired)
