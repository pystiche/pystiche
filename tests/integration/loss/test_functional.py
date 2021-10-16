import pytorch_testing_utils as ptu

import torch
from torch.nn.functional import mse_loss

import pystiche.loss.functional as F


class TestMRFLoss:
    def test_main(self):
        torch.manual_seed(0)
        zero_patch = torch.zeros(4, 5, 6)
        one_patch = torch.ones(4, 5, 6)
        rand_patch = torch.randn(4, 5, 6)

        input = torch.stack((rand_patch + 0.1, rand_patch * 0.9)).unsqueeze(0)
        target = torch.stack((zero_patch, one_patch, rand_patch)).unsqueeze(0)

        actual = F.mrf_loss(input, target, batched_input=True)
        desired = mse_loss(input, torch.stack((rand_patch, rand_patch)).unsqueeze(0))
        ptu.assert_allclose(actual, desired)

    def test_non_batched(self):
        torch.manual_seed(0)
        zero_patch = torch.zeros(4, 5, 6)
        one_patch = torch.ones(4, 5, 6)
        rand_patch = torch.randn(4, 5, 6)

        input = torch.stack((rand_patch + 0.1, rand_patch * 0.9))
        target = torch.stack((zero_patch, one_patch, rand_patch))

        actual = F.mrf_loss(input, target, batched_input=False)
        desired = mse_loss(input, torch.stack((rand_patch, rand_patch)))
        ptu.assert_allclose(actual, desired)


def test_value_range_loss_zero():
    torch.manual_seed(0)
    input = torch.rand(1, 3, 128, 128)

    actual = F.value_range_loss(input)
    desired = 0.0
    assert desired == ptu.approx(actual)


def test_total_variation_loss():
    def get_checkerboard(size):
        return (
            (
                torch.arange(size ** 2).view(size, size)
                + torch.arange(size).view(size, 1)
            )
            % 2
        ).bool()

    size = 128
    checkerboard = get_checkerboard(size)
    input = checkerboard.float().view(1, 1, size, size).repeat(1, 3, 1, 1)

    actual = F.total_variation_loss(input)
    desired = 2.0
    assert desired == ptu.approx(actual)
