import itertools
from collections import OrderedDict

import pytest
import pytorch_testing_utils as ptu

import torch

import pystiche
from pystiche.misc import build_complex_obj_repr

from tests.utils import skip_if_cuda_not_available


def test_ComplexObject_repr_smoke():
    class TestObject(pystiche.ComplexObject):
        pass

    test_object = TestObject()
    assert isinstance(repr(test_object), str)


def test_ComplexObject_repr():
    _properties = OrderedDict((("a", 1),))
    extra_properties = OrderedDict((("b", 2),))
    _named_children = (("c", 3),)
    extra_named_children = (("d", 4),)

    class TestObject(pystiche.ComplexObject):
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

    actual = repr(test_object)
    desired = build_complex_obj_repr(
        name="TestObject", properties=properties, named_children=named_children
    )
    assert actual == desired


def test_LossDict_setitem_Tensor():
    name = "loss"
    loss_dict = pystiche.LossDict()

    loss = torch.tensor(1.0)
    loss_dict[name] = loss

    actual = loss_dict[name]
    desired = loss
    ptu.assert_allclose(actual, desired)


def test_LossDict_setitem_non_scalar_Tensor():
    name = "loss"
    loss_dict = pystiche.LossDict()

    with pytest.raises(TypeError):
        loss_dict[name] = torch.ones(1)


def test_LossDict_setitem_LossDict():
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
        ptu.assert_allclose(actual, desired)


def test_LossDict_setitem_other():
    name = "loss"
    loss_dict = pystiche.LossDict()

    with pytest.raises(TypeError):
        loss_dict[name] = 1.0


def test_LossDict_aggregate_max_depth_gt_0():
    def loss():
        return torch.tensor(1.0)

    loss_dict = pystiche.LossDict(
        (("0.0.0", loss()), ("0.0.1", loss()), ("0.1", loss()), ("1", loss()))
    )

    actual = loss_dict.aggregate(1)
    desired = pystiche.LossDict((("0", 3 * loss()), ("1", loss())))
    ptu.assert_allclose(actual, desired)

    actual = loss_dict.aggregate(2)
    desired = pystiche.LossDict((("0.0", 2 * loss()), ("0.1", loss()), ("1", loss())))
    ptu.assert_allclose(actual, desired)

    actual = loss_dict.aggregate(3)
    desired = loss_dict
    ptu.assert_allclose(actual, desired)

    actual = loss_dict.aggregate(4)
    desired = loss_dict
    ptu.assert_allclose(actual, desired)


def test_LossDict_total():
    loss1 = torch.tensor(1.0)
    loss2 = torch.tensor(2.0)

    loss_dict = pystiche.LossDict((("loss1", loss1), ("loss2", loss2)))

    actual = loss_dict.total()
    desired = loss1 + loss2
    ptu.assert_allclose(actual, desired)


def test_LossDict_backward():
    losses = [
        torch.tensor(val, dtype=torch.float, requires_grad=True) for val in range(3)
    ]

    def zero_grad():
        for loss in losses:
            loss.grad = None

    def extract_grads():
        return [loss.grad.clone() for loss in losses]

    zero_grad()
    loss_dict = pystiche.LossDict([(str(idx), loss) for idx, loss in enumerate(losses)])
    loss_dict.backward()
    actuals = extract_grads()

    zero_grad()
    total = sum(losses)
    total.backward()
    desireds = extract_grads()

    for actual, desired in zip(actuals, desireds):
        ptu.assert_allclose(actual, desired)


def test_LossDict_item():
    losses = (1.0, 2.0)

    loss_dict = pystiche.LossDict(
        [(f"loss{idx}", torch.tensor(val)) for idx, val in enumerate(losses)]
    )

    actual = loss_dict.item()
    desired = sum(losses)
    assert actual == pytest.approx(desired)


def test_LossDict_float():
    loss_dict = pystiche.LossDict((("a", torch.tensor(0.0)), ("b", torch.tensor(1.0))))
    assert float(loss_dict) == pytest.approx(loss_dict.item())


def test_LossDict_mul():
    losses = (2.0, 3.0)
    factor = 4.0

    loss_dict = pystiche.LossDict(
        [(f"loss{idx}", torch.tensor(val)) for idx, val in enumerate(losses)]
    )
    loss_dict = loss_dict * factor

    for idx, loss in enumerate(losses):
        actual = float(loss_dict[f"loss{idx}"])
        desired = loss * factor
        assert actual == ptu.approx(desired)


def test_LossDict_repr_smoke():
    loss_dict = pystiche.LossDict((("a", torch.tensor(0.0)), ("b", torch.tensor(1.0))))
    assert isinstance(repr(loss_dict), str)


def test_TensorKey_eq():
    x = torch.tensor((0.0, 0.5, 1.0))
    key = pystiche.TensorKey(x)

    assert key == key
    assert key == pystiche.TensorKey(x.flip(0))


def test_TensorKey_eq_precision():
    x = torch.tensor(1.0)
    y = torch.tensor(1.0001)

    assert pystiche.TensorKey(x) != pystiche.TensorKey(y)
    assert pystiche.TensorKey(x, precision=3) == pystiche.TensorKey(y, precision=3)


def test_TensorKey_eq_tensor():
    x = torch.tensor((0.0, 0.5, 1.0))
    key = pystiche.TensorKey(x)

    assert key == x


@skip_if_cuda_not_available
def test_TensorKey_eq_device():
    x = torch.tensor((0.0, 0.5, 1.0))

    key1 = pystiche.TensorKey(x.cpu())
    key2 = pystiche.TensorKey(x.cuda())
    assert key1 != key2


def test_TensorKey_eq_dtype():
    x = torch.tensor((0.0, 0.5, 1.0))

    key1 = pystiche.TensorKey(x.float())
    key2 = pystiche.TensorKey(x.double())
    assert key1 != key2


def test_TensorKey_eq_size():
    x = torch.tensor((0.0, 0.5, 1.0))

    key1 = pystiche.TensorKey(x)
    key2 = pystiche.TensorKey(x[:-1])
    assert key1 != key2


def test_TensorKey_eq_min():
    x = torch.tensor((0.0, 0.5, 1.0))

    # This creates a tensor with given min and the same max and norm values as x
    min = 0.1
    intermediate = torch.sqrt(torch.norm(x) ** 2.0 - (1.0 + min ** 2.0)).item()
    y = torch.tensor((min, intermediate, 1.0))

    key1 = pystiche.TensorKey(x)
    key2 = pystiche.TensorKey(y)
    assert key1 != key2


def test_TensorKey_eq_max():
    x = torch.tensor((0.0, 0.5, 1.0))

    # This creates a tensor with given max and the same min and norm values as x
    max = 0.9
    intermediate = torch.sqrt(torch.norm(x) ** 2.0 - max ** 2.0).item()
    y = torch.tensor((0.0, intermediate, max))

    key1 = pystiche.TensorKey(x)
    key2 = pystiche.TensorKey(y)
    assert key1 != key2


def test_TensorKey_eq_norm():
    x = torch.tensor((0.0, 0.5, 1.0))
    y = torch.tensor((0.0, 0.6, 1.0))

    key1 = pystiche.TensorKey(x)
    key2 = pystiche.TensorKey(y)
    assert key1 != key2


def test_TensorKey_hash_smoke():
    x = torch.tensor((0.0, 0.5, 1.0))
    key = pystiche.TensorKey(x)

    assert isinstance(hash(key), int)


def test_TensorKey_repr_smoke():
    x = torch.tensor((0.0, 0.5, 1.0))
    key = pystiche.TensorKey(x)

    assert isinstance(repr(key), str)
