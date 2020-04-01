import os
from os import path
from collections import OrderedDict
import itertools
import tempfile
import torch
from torch import nn
import pystiche
from unittest import mock
from utils import PysticheTestCase


class TestCase(PysticheTestCase):
    def test_Object_str(self):
        _properties = OrderedDict((("a", 1),))
        extra_properties = OrderedDict((("b", 2),))
        _named_children = (("c", 3),)
        extra_named_children = (("d", 4),)

        class TestObject(pystiche.Object):
            def _properties(self):
                return _properties

            def extra_properties(self):
                return extra_properties

            def _named_children(self):
                return iter(_named_children)

            def extra_named_children(self):
                return iter(extra_named_children)

        test_object = TestObject()
        properties = OrderedDict(
            [
                property
                for property in itertools.chain(
                    _properties.items(), extra_properties.items()
                )
            ]
        )
        named_children = tuple(itertools.chain(_named_children, extra_named_children))

        actual = str(test_object)
        desired = test_object._build_str(
            name="TestObject", properties=properties, named_children=named_children
        )
        self.assertEqual(actual, desired)

    def test_LossDict_setitem_Tensor(self):
        name = "loss"
        loss_dict = pystiche.LossDict()

        loss = torch.tensor(1.0)
        loss_dict[name] = loss

        actual = loss_dict[name]
        desired = loss
        self.assertTensorAlmostEqual(actual, desired)

    def test_LossDict_setitem_non_scalar_Tensor(self):
        name = "loss"
        loss_dict = pystiche.LossDict()

        with self.assertRaises(TypeError):
            loss_dict[name] = torch.ones(1)

    def test_LossDict_setitem_LossDict(self):
        name = "loss"
        loss_dict = pystiche.LossDict()
        num_sub_losses = 3

        loss = pystiche.LossDict(
            [(str(idx), torch.tensor(idx)) for idx in range(num_sub_losses)]
        )
        loss_dict[name] = loss

        for idx in range(num_sub_losses):
            actual = loss_dict[f"{name}.{idx}"]
            desired = loss[str(idx)]
            self.assertTensorAlmostEqual(actual, desired)

    def test_LossDict_setitem_other(self):
        name = "loss"
        loss_dict = pystiche.LossDict()

        with self.assertRaises(TypeError):
            loss_dict[name] = 1.0

    def test_LossDict_total(self):
        loss1 = torch.tensor(1.0)
        loss2 = torch.tensor(2.0)

        loss_dict = pystiche.LossDict((("loss1", loss1), ("loss2", loss2)))

        actual = loss_dict.total()
        desired = loss1 + loss2
        self.assertTensorAlmostEqual(actual, desired)

    def test_LossDict_backward(self):
        losses = [
            torch.tensor(val, dtype=torch.float, requires_grad=True) for val in range(3)
        ]

        def zero_grad():
            for loss in losses:
                loss.grad = None

        def extract_grads():
            return [loss.grad.clone() for loss in losses]

        zero_grad()
        loss_dict = pystiche.LossDict(
            [(str(idx), loss) for idx, loss in enumerate(losses)]
        )
        loss_dict.backward()
        actuals = extract_grads()

        zero_grad()
        total = sum(losses)
        total.backward()
        desireds = extract_grads()

        for actual, desired in zip(actuals, desireds):
            self.assertTensorAlmostEqual(actual, desired)

    def test_LossDict_item(self):
        losses = (1.0, 2.0)

        loss_dict = pystiche.LossDict(
            [(f"loss{idx}", torch.tensor(val)) for idx, val in enumerate(losses)]
        )

        actual = loss_dict.item()
        desired = sum(losses)
        self.assertAlmostEqual(actual, desired)

    def test_LossDict_float(self):
        loss_dict = pystiche.LossDict(
            (("a", torch.tensor(0.0)), ("b", torch.tensor(1.0)))
        )
        self.assertAlmostEqual(float(loss_dict), loss_dict.item())

    def test_LossDict_mul(self):
        losses = (2.0, 3.0)
        factor = 4.0

        loss_dict = pystiche.LossDict(
            [(f"loss{idx}", torch.tensor(val)) for idx, val in enumerate(losses)]
        )
        loss_dict = loss_dict * factor

        for idx, loss in enumerate(losses):
            actual = float(loss_dict[f"loss{idx}"])
            desired = loss * factor
            self.assertAlmostEqual(actual, desired)

    def test_LossDict_str_smoke(self):
        loss_dict = pystiche.LossDict(
            (("a", torch.tensor(0.0)), ("b", torch.tensor(1.0)))
        )
        self.assertIsInstance(str(loss_dict), str)

    @mock.patch("pystiche.core._home.os.makedirs")
    def test_home_default(self, makedirs_mock):
        actual = pystiche.home()
        desired = path.expanduser(path.join("~", ".cache", "pystiche"))
        self.assertEqual(actual, desired)

    @mock.patch("pystiche.core._home.os.getenv")
    def test_home_env(self, getenv_mock):
        tmp_dir = tempfile.mkdtemp()
        os.rmdir(tmp_dir)
        getenv_mock.return_value = tmp_dir
        try:
            actual = tmp_dir
            desired = pystiche.home()
            self.assertEqual(actual, desired)
            self.assertTrue(path.exists(desired) and path.isdir(desired))
        finally:
            os.rmdir(tmp_dir)

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

    def test_tensor_meta(self):
        meta = {"dtype": torch.bool, "device": torch.device("cpu")}

        x = torch.empty((), **meta)

        actual = pystiche.tensor_meta(x)
        desired = meta
        self.assertDictEqual(actual, desired)

    def test_tensor_meta_kwargs(self):
        dtype = torch.bool

        x = torch.empty(())

        actual = pystiche.tensor_meta(x, dtype=dtype)["dtype"]
        desired = dtype
        self.assertEqual(actual, desired)

    def test_is_scalar_tensor(self):
        for scalar_tensor in (torch.tensor(0.0), torch.empty(())):
            self.assertTrue(pystiche.is_scalar_tensor(scalar_tensor))

        for nd_tensor in (torch.empty(0), torch.empty((0,))):
            self.assertFalse(pystiche.is_scalar_tensor(nd_tensor))

    def test_conv_module_meta(self):
        meta = {
            "kernel_size": (2,),
            "stride": (3,),
            "padding": (4,),
            "dilation": (5,),
        }

        x = nn.Conv1d(1, 1, **meta)

        actual = pystiche.conv_module_meta(x)
        desired = meta
        self.assertDictEqual(actual, desired)

    def test_conv_module_meta_kwargs(self):
        stride = (2,)

        x = nn.Conv1d(1, 1, 1, stride=stride)

        actual = pystiche.conv_module_meta(x, stride=stride)["stride"]
        desired = stride
        self.assertEqual(actual, desired)

    def test_pool_module_meta(self):
        meta = {
            "kernel_size": (2,),
            "stride": (3,),
            "padding": (4,),
        }

        x = nn.MaxPool1d(**meta)

        actual = pystiche.pool_module_meta(x)
        desired = meta
        self.assertDictEqual(actual, desired)

    def test_pool_module_meta_kwargs(self):
        kernel_size = (2,)

        x = nn.MaxPool1d(kernel_size=kernel_size)

        actual = pystiche.conv_module_meta(x, kernel_size=kernel_size)["kernel_size"]
        desired = kernel_size
        self.assertEqual(actual, desired)
