from math import sqrt

import torch

import pystiche


def test_nonnegsqrt(self):
    vals = (-1.0, 0.0, 1.0, 2.0)
    desireds = (0.0, 0.0, 1.0, sqrt(2.0))

    for val, desired in zip(vals, desireds):
        x = torch.tensor(val, requires_grad=True)
        y = pystiche.nonnegsqrt(x)

        actual = y.item()
        self.assertAlmostEqual(actual, desired)


def test_nonnegsqrt_grad(self):
    vals = (-1.0, 0.0, 1.0, 2.0)
    desireds = (0.0, 0.0, 1.0 / 2.0, 1.0 / (2.0 * sqrt(2.0)))

    for val, desired in zip(vals, desireds):
        x = torch.tensor(val, requires_grad=True)
        y = pystiche.nonnegsqrt(x)
        y.backward()

        actual = x.grad.item()
        self.assertAlmostEqual(actual, desired)


def test_gram_matrix(self):
    size = 100

    for dim in (1, 2, 3):
        x = torch.ones((1, 1, *[size] * dim))
        y = pystiche.gram_matrix(x)

        actual = y.item()
        desired = float(size ** dim)
        self.assertAlmostEqual(actual, desired)


def test_gram_matrix_size(self):
    batch_size = 1
    num_channels = 3

    torch.manual_seed(0)
    for dim in (1, 2, 3):
        size = (batch_size, num_channels, *torch.randint(256, (dim,)).tolist())
        x = torch.empty(size)
        y = pystiche.gram_matrix(x)

        actual = y.size()
        desired = (batch_size, num_channels, num_channels)
        self.assertTupleEqual(actual, desired)


def test_gram_matrix_normalize1(self):
    num_channels = 3

    x = torch.ones((1, num_channels, 128, 128))
    y = pystiche.gram_matrix(x, normalize=True)

    actual = y.flatten()
    desired = torch.ones((num_channels ** 2,))
    self.assertTensorAlmostEqual(actual, desired)


def test_gram_matrix_normalize2(self):
    torch.manual_seed(0)
    tensor_constructors = (torch.ones, torch.rand, torch.randn)

    for constructor in tensor_constructors:
        x = pystiche.gram_matrix(constructor((1, 3, 128, 128)), normalize=True)
        y = pystiche.gram_matrix(constructor((1, 3, 256, 256)), normalize=True)

        self.assertTensorAlmostEqual(x, y, atol=2e-2)
