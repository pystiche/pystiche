import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

import pystiche


class TestModule:
    def test_core(self):
        class EmptyModule(pystiche.Module):
            def forward(self):
                pass

        childs = (nn.Conv2d(1, 1, 1), nn.ReLU())
        named_children = [(f"child{idx}", child) for idx, child in enumerate(childs)]
        indexed_children = childs

        test_module = EmptyModule(named_children=named_children)
        for idx, child in enumerate(childs):
            actual = getattr(test_module, f"child{idx}")
            desired = child
            assert actual is desired

        test_module = EmptyModule(indexed_children=indexed_children)
        for idx, child in enumerate(childs):
            actual = getattr(test_module, str(idx))
            desired = child
            assert actual is desired

        with pytest.raises(RuntimeError):
            EmptyModule(
                named_children=named_children, indexed_children=indexed_children
            )

    def test_repr_smoke(self):
        class EmptyModule(pystiche.Module):
            def forward(self):
                pass

        test_module = EmptyModule()
        assert isinstance(repr(test_module), str)

    def test_torch_repr_smoke(self):
        class EmptyModule(pystiche.Module):
            def forward(self):
                pass

        test_module = EmptyModule()
        assert isinstance(test_module.torch_repr(), str)


class TestSequentialModule:
    def test_core(self):
        modules = (nn.Conv2d(3, 3, 3), nn.ReLU())
        model = pystiche.SequentialModule(*modules)

        for idx, module in enumerate(modules):
            actual = getattr(model, str(idx))
            desired = module
            assert actual is desired

    def test_call(self):
        torch.manual_seed(0)
        modules = (nn.Conv2d(3, 3, 3), nn.ReLU())
        input = torch.rand(1, 3, 256, 256)

        pystiche_model = pystiche.SequentialModule(*modules)
        torch_model = nn.Sequential(*modules)

        actual = pystiche_model(input)
        desired = torch_model(input)
        ptu.assert_allclose(actual, desired)
