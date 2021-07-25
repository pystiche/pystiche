import pytorch_testing_utils as ptu

import torch

import pystiche.loss.functional as F
from pystiche import loss


class TestTotalVariationLoss:
    def test_call(self):
        torch.manual_seed(0)
        image = torch.rand(1, 3, 128, 128)
        exponent = 3.0

        op = loss.TotalVariationLoss(exponent=exponent)

        actual = op(image)
        desired = F.total_variation_loss(image, exponent=exponent)
        ptu.assert_allclose(actual, desired)

    def test_repr_smoke(self, encoder):
        assert isinstance(repr(loss.TotalVariationLoss()), str)


class TestValueRangeLoss:
    def test_call(self):
        torch.manual_seed(0)
        image = torch.randn(1, 3, 128, 128)

        op = loss.ValueRangeLoss()

        actual = op(image)
        desired = F.value_range_loss(image)
        ptu.assert_allclose(actual, desired)

    def test_repr_smoke(self, encoder):
        assert isinstance(repr(loss.ValueRangeLoss()), str)
