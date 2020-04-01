import os
from os import path
import tempfile
import torch
import pystiche
from unittest import mock
from utils import PysticheTestCase


class TestCase(PysticheTestCase):
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
