import itertools
import os
import tempfile
from collections import OrderedDict
from math import sqrt
from os import path
from unittest import mock

import torch
from torch import nn

import pystiche
from utils import PysticheTestCase, skip_if_cuda_not_available


class TestBase(PysticheTestCase):
    def test_Object_str_smoke(self):
        class TestObject(pystiche.Object):
            pass

        test_object = TestObject()
        self.assertIsInstance(str(test_object), str)

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

    def test_LossDict_aggregate_max_depth_gt_0(self):
        def loss():
            return torch.tensor(1.0)

        loss_dict = pystiche.LossDict(
            (("0.0.0", loss()), ("0.0.1", loss()), ("0.1", loss()), ("1", loss()))
        )

        actual = loss_dict.aggregate(1)
        desired = pystiche.LossDict((("0", 3 * loss()), ("1", loss())))
        self.assertDictEqual(actual, desired)

        actual = loss_dict.aggregate(2)
        desired = pystiche.LossDict(
            (("0.0", 2 * loss()), ("0.1", loss()), ("1", loss()))
        )
        self.assertDictEqual(actual, desired)

        actual = loss_dict.aggregate(3)
        desired = loss_dict
        self.assertDictEqual(actual, desired)

        actual = loss_dict.aggregate(4)
        desired = loss_dict
        self.assertDictEqual(actual, desired)

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

    def test_TensorKey_eq(self):
        x = torch.tensor((0.0, 0.5, 1.0))
        key = pystiche.TensorKey(x)

        self.assertTrue(key == key)
        self.assertTrue(key == pystiche.TensorKey(x.flip(0)))

    def test_TensorKey_eq_tensor(self):
        x = torch.tensor((0.0, 0.5, 1.0))
        key = pystiche.TensorKey(x)

        self.assertTrue(key == x)

    def test_TensorKey_eq_other(self):
        x = torch.tensor((0.0, 0.5, 1.0))
        key = pystiche.TensorKey(x)

        with self.assertRaises(TypeError):
            key == 1

    @skip_if_cuda_not_available
    def test_TensorKey_eq_device(self):
        x = torch.tensor((0.0, 0.5, 1.0))

        key1 = pystiche.TensorKey(x.cpu())
        key2 = pystiche.TensorKey(x.cuda())
        self.assertFalse(key1 == key2)

    def test_TensorKey_eq_dtype(self):
        x = torch.tensor((0.0, 0.5, 1.0))

        key1 = pystiche.TensorKey(x.float())
        key2 = pystiche.TensorKey(x.double())
        self.assertFalse(key1 == key2)

    def test_TensorKey_eq_size(self):
        x = torch.tensor((0.0, 0.5, 1.0))

        key1 = pystiche.TensorKey(x)
        key2 = pystiche.TensorKey(x[:-1])
        self.assertFalse(key1 == key2)

    def test_TensorKey_eq_min(self):
        x = torch.tensor((0.0, 0.5, 1.0))

        # This creates a tensor with given min and the same max and norm values as x
        min = 0.1
        intermediate = torch.sqrt(torch.norm(x) ** 2.0 - (1.0 + min ** 2.0)).item()
        y = torch.tensor((min, intermediate, 1.0))

        key1 = pystiche.TensorKey(x)
        key2 = pystiche.TensorKey(y)
        self.assertFalse(key1 == key2)

    def test_TensorKey_eq_max(self):
        x = torch.tensor((0.0, 0.5, 1.0))

        # This creates a tensor with given max and the same min and norm values as x
        max = 0.9
        intermediate = torch.sqrt(torch.norm(x) ** 2.0 - max ** 2.0).item()
        y = torch.tensor((0.0, intermediate, max))

        key1 = pystiche.TensorKey(x)
        key2 = pystiche.TensorKey(y)
        self.assertFalse(key1 == key2)

    def test_TensorKey_eq_norm(self):
        x = torch.tensor((0.0, 0.5, 1.0))
        y = torch.tensor((0.0, 0.6, 1.0))

        key1 = pystiche.TensorKey(x)
        key2 = pystiche.TensorKey(y)
        self.assertFalse(key1 == key2)

    def test_TensorKey_hash_smoke(self):
        x = torch.tensor((0.0, 0.5, 1.0))
        key = pystiche.TensorKey(x)

        self.assertIsInstance(hash(key), int)


class TestHome(PysticheTestCase):
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


class TestMath(PysticheTestCase):
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

    def test_batch_gram_matrix(self):
        size = 100

        for dim in (1, 2, 3):
            x = torch.ones((1, 1, *[size] * dim))
            y = pystiche.batch_gram_matrix(x)

            actual = y.item()
            desired = float(size ** dim)
            self.assertAlmostEqual(actual, desired)

    def test_batch_gram_matrix_size(self):
        batch_size = 1
        num_channels = 3

        torch.manual_seed(0)
        for dim in (1, 2, 3):
            size = (batch_size, num_channels, *torch.randint(256, (dim,)).tolist())
            x = torch.empty(size)
            y = pystiche.batch_gram_matrix(x)

            actual = y.size()
            desired = (batch_size, num_channels, num_channels)
            self.assertTupleEqual(actual, desired)

    def test_batch_gram_matrix_normalize1(self):
        num_channels = 3

        x = torch.ones((1, num_channels, 128, 128))
        y = pystiche.batch_gram_matrix(x, normalize=True)

        actual = y.flatten()
        desired = torch.ones((num_channels ** 2,))
        self.assertTensorAlmostEqual(actual, desired)

    def test_batch_gram_matrix_normalize2(self):
        torch.manual_seed(0)
        tensor_constructors = (torch.ones, torch.rand, torch.randn)

        for constructor in tensor_constructors:
            x = pystiche.batch_gram_matrix(
                constructor((1, 3, 128, 128)), normalize=True
            )
            y = pystiche.batch_gram_matrix(
                constructor((1, 3, 256, 256)), normalize=True
            )

            self.assertTensorAlmostEqual(x, y, atol=2e-2)


class TestMeta(PysticheTestCase):
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


class TestModules(PysticheTestCase):
    def test_Module(self):
        class TestModule(pystiche.Module):
            def forward(self):
                pass

        childs = (nn.Conv2d(1, 1, 1), nn.ReLU())
        named_children = OrderedDict(
            [(f"child{idx}", child) for idx, child in enumerate(childs)]
        )
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

    def test_Module_extra_repr_smoke(self):
        class TestModule(pystiche.Module):
            def forward(self):
                pass

        test_module = TestModule()
        self.assertIsInstance(test_module.extra_repr(), str)

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


class TestUtils(PysticheTestCase):
    def test_extract_patchesnd_num_patches(self):
        batch_size = 2
        size = 64
        patch_size = 3
        stride = 2

        num_patches_per_dim = int((size - patch_size) // stride + 1)
        torch.manual_seed(0)
        for dims, extract_patches in enumerate(
            (
                pystiche.extract_patches1d,
                pystiche.extract_patches2d,
                pystiche.extract_patches3d,
            ),
            1,
        ):
            x = torch.rand(batch_size, 1, *[size] * dims)
            patches = extract_patches(x, patch_size, stride=stride)

            actual = patches.size()[0]
            desired = batch_size * num_patches_per_dim ** dims
            self.assertEqual(actual, desired)

    def test_extract_patches1d(self):
        batch_size = 2
        length = 9
        patch_size = 3
        stride = 2

        x = torch.arange(batch_size * length).view(batch_size, 1, -1)
        patches = pystiche.extract_patches1d(x, patch_size, stride=stride)

        actual = patches[0]
        desired = x[0, :, :patch_size]
        self.assertTensorAlmostEqual(actual, desired)

        actual = patches[1]
        desired = x[0, :, stride : (stride + patch_size)]
        self.assertTensorAlmostEqual(actual, desired)

        actual = patches[-1]
        desired = x[-1, :, -patch_size:]
        self.assertTensorAlmostEqual(actual, desired)
