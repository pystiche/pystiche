from math import sqrt

import pytorch_testing_utils as ptu

import torch
import torch.nn.functional as F

import pystiche


class TestNonNegSqrt:
    def test_main(self):
        vals = (-1.0, 0.0, 1.0, 2.0)
        desireds = (0.0, 0.0, 1.0, sqrt(2.0))

        for val, desired in zip(vals, desireds):
            x = torch.tensor(val)
            y = pystiche.nonnegsqrt(x)

            assert y == ptu.approx(desired)

    def test_grad(self):
        vals = (-1.0, 0.0, 1.0, 2.0)
        desireds = (0.0, 0.0, 1.0 / 2.0, 1.0 / (2.0 * sqrt(2.0)))

        for val, desired in zip(vals, desireds):
            x = torch.tensor(val, requires_grad=True)
            y = pystiche.nonnegsqrt(x)
            y.backward()

            assert x.grad == ptu.approx(desired)


class TestGramMatrix:
    def test_main(self):
        size = 100

        for dim in (1, 2, 3):
            x = torch.ones((1, 1, *[size] * dim))
            y = pystiche.gram_matrix(x)

            actual = y.item()
            desired = float(size ** dim)
            assert actual == desired

    def test_size(self):
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

    def test_normalize1(self):
        num_channels = 3

        x = torch.ones((1, num_channels, 128, 128))
        y = pystiche.gram_matrix(x, normalize=True)

        actual = y.flatten()
        desired = torch.ones((num_channels ** 2,))
        ptu.assert_allclose(actual, desired)

    def test_normalize2(self):
        torch.manual_seed(0)
        tensor_constructors = (torch.ones, torch.rand, torch.randn)

        for constructor in tensor_constructors:
            x = pystiche.gram_matrix(constructor((1, 3, 128, 128)), normalize=True)
            y = pystiche.gram_matrix(constructor((1, 3, 256, 256)), normalize=True)

            ptu.assert_allclose(x, y, atol=2e-2)


class TestCosineSimilarity:
    def test_shape(self):
        torch.manual_seed(0)
        input = torch.rand(2, 3, 4, 5)
        target = torch.rand(2, 3, 4, 5)

        assert pystiche.cosine_similarity(input, target).size() == (2, 3, 3)

    def test_input(self):
        torch.manual_seed(0)
        x1 = torch.rand(2, 1, 256)
        x2 = torch.rand(2, 1, 256)
        eps = 1e-6

        actual = pystiche.cosine_similarity(x1, x2, eps=eps, batched_input=True)
        expected = F.cosine_similarity(x1, x2, dim=2, eps=eps).unsqueeze(2)
        ptu.assert_allclose(actual, expected, rtol=1e-6)

    def test_non_batched(self):
        torch.manual_seed(0)
        input = torch.rand(1, 256)
        target = torch.rand(1, 256)
        eps = 1e-6

        actual = pystiche.cosine_similarity(input, target, eps=eps, batched_input=False)
        expected = F.cosine_similarity(input, target, dim=1, eps=eps).unsqueeze(1)
        ptu.assert_allclose(actual, expected, rtol=1e-6)
