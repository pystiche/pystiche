import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

from pystiche import enc

from tests import mocks


@pytest.fixture
def mle_and_modules(module_factory):
    shallow = module_factory()
    intermediate = module_factory()
    deep = module_factory()
    modules = [("shallow", shallow), ("intermediate", intermediate), ("deep", deep)]
    mle = enc.MultiLayerEncoder(modules)
    return mle, dict(modules)


@pytest.fixture
def mle(mle_and_modules):
    return mle_and_modules[0]


@pytest.fixture
def input():
    torch.manual_seed(0)
    return torch.rand(1, 3, 16, 16)


def test_MultiLayerEncoder_contains(mle):
    assert "shallow" in mle
    assert "intermediate" in mle
    assert "deep" in mle
    assert "unknown" not in mle


def test_MultiLayerEncoder_register_layer(mle):
    layer = "shallow"

    assert layer not in mle.registered_layers

    mle.register_layer(layer)

    assert layer in mle.registered_layers


def test_MultiLayerEncoder_register_layer_error(mle):
    with pytest.raises(ValueError):
        mle.register_layer("unknown")


def test_MultiLayerEncoder_call(input):
    conv = nn.Conv2d(3, 1, 1)
    relu = nn.ReLU(inplace=False)

    modules = (("conv", conv), ("relu", relu))
    mle = enc.MultiLayerEncoder(modules)

    actual = mle(input, "conv")
    expected = conv(input)
    ptu.assert_allclose(actual, expected)

    actual = mle(input, "relu")
    expected = relu(conv(input))
    ptu.assert_allclose(actual, expected)


def test_MultiLayerEncoder_call_default_layer(mle_and_modules, input):
    mle, modules = mle_and_modules

    mle(input)
    modules["deep"].assert_called_once_with(input)


def test_MultiLayerEncoder_call_error(mle):
    with pytest.raises(ValueError):
        mle(torch.empty(1), "unknown")


def test_MultiLayerEncoder_call_use_cached(mle_and_modules, input):
    mle, modules = mle_and_modules
    layer = "deep"
    mle.register_layer(layer)

    mle(input, layer)
    modules[layer].assert_called_once_with(input)

    mle(input, layer)
    modules[layer].assert_called_once()


def test_MultiLayerEncoder_call_resume_from_cache(mle_and_modules, input):
    mle, modules = mle_and_modules
    mle.register_layer("shallow")

    mle(input, "shallow")
    modules["shallow"].assert_called_once_with(input)

    mle(input, "deep")
    modules["shallow"].assert_called_once()
    modules["deep"].assert_called_once_with(input)


def test_MultiLayerEncoder_call_pre_cache(mle_and_modules, input):
    mle, modules = mle_and_modules
    mle.register_layer("shallow")

    mle(input, "deep")
    modules["shallow"].assert_called_once_with(input)
    modules["deep"].assert_called_once_with(input)

    mle(input, "shallow")
    modules["shallow"].assert_called_once()


def test_MultiLayerEncoder_clear_cache(mle_and_modules, input):
    mle, modules = mle_and_modules
    layer = "shallow"
    module = modules[layer]
    mle.register_layer(layer)

    mle(input, layer)
    module.assert_called_once_with(input)

    mle.clear_cache()

    mle(input, layer)
    module.assert_called(2)
    module.assert_called_with(input)


def test_MultiLayerEncoder_empty_storage(mocker, mle):
    mock = mocker.patch(
        mocks.make_mock_target(
            "enc", "multi_layer_encoder", "MultiLayerEncoder", "clear_cache"
        )
    )

    with pytest.warns(UserWarning):
        mle.empty_storage()

    mock.assert_called()


def test_MultiLayerEncoder_encode(input):
    conv = nn.Conv2d(3, 1, 1)
    relu = nn.ReLU(inplace=False)

    modules = (("conv", conv), ("relu", relu))
    mle = enc.MultiLayerEncoder(modules)

    encs = mle.encode(input, ("conv", "relu"))

    actual = encs[0]
    expected = conv(input)
    ptu.assert_allclose(actual, expected)

    actual = encs[1]
    expected = relu(conv(input))
    ptu.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "layers",
    (("shallow", "deep"), ("deep", "shallow")),
    ids=lambda layers: f"{layers[0]} to {layers[1]}",
)
def test_MultiLayerEncoder_encode_cache(mle_and_modules, input, layers):
    mle, modules = mle_and_modules

    mle.encode(input, layers)

    modules["shallow"].assert_called_once_with(input)
    modules["deep"].assert_called_once_with(input)


def test_MultiLayerEncoder_trim(mle):
    assert "shallow" in mle
    assert "intermediate" in mle
    assert "deep" in mle

    mle.trim(("shallow",))

    assert "shallow" in mle
    assert "intermediate" not in mle
    assert "deep" not in mle


def test_MultiLayerEncoder_trim_registered(mle):
    mle.register_layer("shallow")

    assert "shallow" in mle
    assert "intermediate" in mle
    assert "deep" in mle

    mle.trim()

    assert "shallow" in mle
    assert "intermediate" not in mle
    assert "deep" not in mle


def test_MultiLayerEncoder_extract_encoder(mle):
    layer = "intermediate"
    encoder = mle.extract_encoder(layer)

    assert isinstance(encoder, enc.SingleLayerEncoder)
    assert encoder.multi_layer_encoder is mle
    assert encoder.layer == layer
    assert layer in mle.registered_layers


def test_SingleLayerEncoder_call(input):
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 1, 1)
    relu = nn.ReLU(inplace=False)

    modules = (("conv", conv), ("relu", relu))
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    single_layer_encoder = enc.SingleLayerEncoder(multi_layer_encoder, "relu")

    actual = single_layer_encoder(input)
    expected = relu(conv(input))
    ptu.assert_allclose(actual, expected)
