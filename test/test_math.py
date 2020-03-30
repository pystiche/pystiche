from math import sqrt
import torch
import pystiche
from utils import PysticheTestCase


class TestCase(PysticheTestCase):
    def test_possqrt(self):
        vals = (-1.0, 0.0, 1.0, 2.0)
        desireds = (0.0, 0.0, 1.0, sqrt(2.0))

        for val, desired in zip(vals, desireds):
            x = torch.tensor(val, requires_grad=True)
            y = pystiche.possqrt(x)

            actual = y.item()
            self.assertAlmostEqual(actual, desired)

    def test_possqrt_grad(self):
        vals = (-1.0, 0.0, 1.0, 2.0)
        desireds = (0.0, 0.0, 1.0 / 2.0, 1.0 / (2.0 * sqrt(2.0)))

        for val, desired in zip(vals, desireds):
            x = torch.tensor(val, requires_grad=True)
            y = pystiche.possqrt(x)
            y.backward()

            actual = x.grad.item()
            self.assertAlmostEqual(actual, desired)
