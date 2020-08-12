import math
from os import path

import pytest
import pytorch_testing_utils as ptu

import torch

from pystiche import misc
from pystiche.image import read_image


def test_prod():
    n = 10
    iterable = range(1, n + 1)

    actual = misc.prod(iterable)
    desired = math.factorial(n)
    assert actual == desired


def test_to_1d_arg():
    val = 0
    actual = misc.to_1d_arg(val)
    desired = (val,)
    assert actual == desired

    val = (0,)
    actual = misc.to_1d_arg(val)
    desired = val
    assert actual == desired

    val = 0
    actual = misc.to_1d_arg([val])
    desired = (val,)
    assert actual == desired

    val = (0, 0)
    with pytest.raises(RuntimeError):
        misc.to_1d_arg(val)


def test_to_2d_arg():
    val = 0
    actual = misc.to_2d_arg(val)
    desired = (val, val)
    assert actual == desired

    val = (0, 0)
    actual = misc.to_2d_arg(val)
    desired = val
    assert actual == desired

    val = 0
    actual = misc.to_2d_arg([val] * 2)
    desired = (val, val)
    assert actual == desired

    val = (0,)
    with pytest.raises(RuntimeError):
        misc.to_2d_arg(val)


def test_to_3d_arg():
    val = 0
    actual = misc.to_3d_arg(val)
    desired = (val, val, val)
    assert actual == desired

    val = (0, 0, 0)
    actual = misc.to_3d_arg(val)
    desired = val
    assert actual == desired

    val = 0
    actual = misc.to_3d_arg([val] * 3)
    desired = (val, val, val)
    assert actual == desired

    val = (0,)
    with pytest.raises(RuntimeError):
        misc.to_3d_arg(val)


def test_zip_equal():
    foo = (1, 2)
    bar = ("a", "b")

    actual = tuple(misc.zip_equal(foo, bar))
    desired = tuple(zip(foo, bar))
    assert actual == desired

    foo = (1, 2)
    bar = ("a", "b", "c")

    with pytest.raises(RuntimeError):
        misc.zip_equal(foo, bar)


def test_verify_str_arg():
    arg = None
    with pytest.raises(ValueError):
        misc.verify_str_arg(arg)

    arg = "foo"
    valid_args = ("bar", "baz")
    with pytest.raises(ValueError):
        misc.verify_str_arg(arg, valid_args=valid_args)

    arg = "foo"
    valid_args = ("foo", "bar")

    actual = misc.verify_str_arg(arg, valid_args=valid_args)
    desired = arg
    assert actual == desired


def test_get_input_image_tensor():
    image = torch.tensor(0.0)

    starting_point = image
    actual = misc.get_input_image(starting_point)
    desired = image
    assert actual is not desired
    ptu.assert_allclose(actual, desired)


def test_get_input_image_tensor_content():
    starting_point = "content"
    image = torch.tensor(0.0)

    actual = misc.get_input_image(starting_point, content_image=image)
    desired = image
    assert actual == ptu.approx(desired)

    with pytest.raises(RuntimeError):
        misc.get_input_image(starting_point, style_image=image)


def test_get_input_image_tensor_style():
    starting_point = "style"
    image = torch.tensor(0.0)

    actual = misc.get_input_image(starting_point, style_image=image)
    desired = image
    assert actual == ptu.approx(desired)

    with pytest.raises(RuntimeError):
        misc.get_input_image(starting_point, content_image=image)


def test_get_input_image_tensor_random():
    starting_point = "random"
    content_image = torch.tensor(0.0, dtype=torch.float32)
    style_image = torch.tensor(0.0, dtype=torch.float64)

    actual = misc.get_input_image(starting_point, content_image=content_image)
    desired = content_image
    ptu.assert_tensor_attributes_equal(actual, desired)

    actual = misc.get_input_image(starting_point, style_image=style_image)
    desired = style_image
    ptu.assert_tensor_attributes_equal(actual, desired)

    actual = misc.get_input_image(
        starting_point, content_image=content_image, style_image=style_image
    )
    desired = content_image
    ptu.assert_tensor_attributes_equal(actual, desired)

    with pytest.raises(RuntimeError):
        misc.get_input_image(starting_point)


def test_get_device():
    actual = misc.get_device()
    desired = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert actual == desired


def test_get_device_str():
    device_name = "mkldnn"
    actual = misc.get_device(device_name)
    desired = torch.device(device_name)
    assert actual == desired


def test_download_file(tmpdir, test_image_url, test_image):
    file = path.join(tmpdir, path.basename(test_image_url))
    misc.download_file(test_image_url, file)

    actual = read_image(file)
    desired = test_image
    ptu.assert_allclose(actual, desired)


def test_reduce():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128)

    actual = misc.reduce(x, "mean")
    desired = torch.mean(x)
    ptu.assert_allclose(actual, desired)

    actual = misc.reduce(x, "sum")
    desired = torch.sum(x)
    ptu.assert_allclose(actual, desired)

    actual = misc.reduce(x, "none")
    desired = x
    ptu.assert_allclose(actual, desired)
