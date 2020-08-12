import torch
from torch import nn

import pystiche


def test_Module(self):
    class TestModule(pystiche.Module):
        def forward(self):
            pass

    childs = (nn.Conv2d(1, 1, 1), nn.ReLU())
    named_children = [(f"child{idx}", child) for idx, child in enumerate(childs)]
    indexed_children = childs

    test_module = TestModule(named_children=named_children)
    for idx, child in enumerate(childs):
        actual = getattr(test_module, f"child{idx}")
        desired = child
        self.assertIs(actual, desired)

    test_module = TestModule(indexed_children=indexed_children)
    for idx, child in enumerate(childs):
        actual = getattr(test_module, str(idx))
        desired = child
        self.assertIs(actual, desired)

    with self.assertRaises(RuntimeError):
        TestModule(named_children=named_children, indexed_children=indexed_children)


def test_Module_repr_smoke(self):
    class TestModule(pystiche.Module):
        def forward(self):
            pass

    test_module = TestModule()
    self.assertIsInstance(repr(test_module), str)


def test_Module_torch_repr_smoke(self):
    class TestModule(pystiche.Module):
        def forward(self):
            pass

    test_module = TestModule()
    self.assertIsInstance(test_module.torch_repr(), str)


def test_SequentialModule(self):
    modules = (nn.Conv2d(3, 3, 3), nn.ReLU())
    model = pystiche.SequentialModule(*modules)

    for idx, module in enumerate(modules):
        actual = getattr(model, str(idx))
        desired = module
        self.assertIs(actual, desired)


def test_SequentialModule_call(self):
    torch.manual_seed(0)
    modules = (nn.Conv2d(3, 3, 3), nn.ReLU())
    input = torch.rand(1, 3, 256, 256)

    pystiche_model = pystiche.SequentialModule(*modules)
    torch_model = nn.Sequential(*modules)

    actual = pystiche_model(input)
    desired = torch_model(input)
    self.assertTensorAlmostEqual(actual, desired)
