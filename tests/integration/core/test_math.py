from math import sqrt

import pytorch_testing_utils as ptu

import torch
import torch.nn.functional as F

import pystiche


def test_nonnegsqrt():
    vals = (-1.0, 0.0, 1.0, 2.0)
    desireds = (0.0, 0.0, 1.0, sqrt(2.0))

    for val, desired in zip(vals, desireds):
        x = torch.tensor(val, requires_grad=True)
        y = pystiche.nonnegsqrt(x)

        assert y == ptu.approx(desired)


def test_nonnegsqrt_grad():
    vals = (-1.0, 0.0, 1.0, 2.0)
    desireds = (0.0, 0.0, 1.0 / 2.0, 1.0 / (2.0 * sqrt(2.0)))

    for val, desired in zip(vals, desireds):
        x = torch.tensor(val, requires_grad=True)
        y = pystiche.nonnegsqrt(x)
        y.backward()

        assert x.grad == ptu.approx(desired)


def test_gram_matrix():
    size = 100

    for dim in (1, 2, 3):
        x = torch.ones((1, 1, *[size] * dim))
        y = pystiche.gram_matrix(x)

        actual = y.item()
        desired = float(size ** dim)
        assert actual == desired


def test_gram_matrix_size():
    batch_size = 1
    num_channels = 3

    torch.manual_seed(0)
    for dim in (1, 2, 3):
        size = (batch_size, num_channels, *torch.randint(256, (dim,)).tolist())
        x = torch.empty(size)
        y = pystiche.gram_matrix(x)

        actual = y.size()
        desired = (batch_size, num_channels, num_channels)
        assert actual == desired


def test_gram_matrix_normalize1():
    num_channels = 3

    x = torch.ones((1, num_channels, 128, 128))
    y = pystiche.gram_matrix(x, normalize=True)

    actual = y.flatten()
    desired = torch.ones((num_channels ** 2,))
    ptu.assert_allclose(actual, desired)


def test_gram_matrix_normalize2():
    torch.manual_seed(0)
    tensor_constructors = (torch.ones, torch.rand, torch.randn)

    for constructor in tensor_constructors:
        x = pystiche.gram_matrix(constructor((1, 3, 128, 128)), normalize=True)
        y = pystiche.gram_matrix(constructor((1, 3, 256, 256)), normalize=True)

        ptu.assert_allclose(x, y, atol=2e-2)


def test_cosine_similarity():
    torch.manual_seed(0)
    input = torch.rand(1, 256)
    target = torch.rand(1, 256)

    actual = pystiche.cosine_similarity(input, target)
    expected = F.cosine_similarity(input, target)
    ptu.assert_allclose(actual, expected)


def test_cosine_similarity_shape():
    torch.manual_seed(0)
    input = torch.rand(2, 3, 4, 5)
    target = torch.rand(2, 3, 4, 5)

    assert pystiche.cosine_similarity(input, target).size() == (2, 2)
