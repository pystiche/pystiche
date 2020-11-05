import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

from pystiche import enc

from tests import mocks


@pytest.fixture
def mle_and_modules(module_factory):
    shallow = module_factory()
    deep = module_factory()
    modules = [("shallow", shallow), ("deep", deep)]
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
    assert "deep" in mle
    assert "unknown" not in mle


def test_MultiLayerEncoder_registered_layer(mle):
    assert not mle.registered_layers


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


def test_MultiLayerEncoder_trim():
    layers = [str(idx) for idx in range(3)]
    modules = [(layer, nn.Module()) for layer in layers]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    for name, module in modules:
        actual = getattr(multi_layer_encoder, name)
        desired = module
        assert actual is desired

    idx = 1
    multi_layer_encoder.trim((str(idx),))

    for name, module in modules[: idx + 1]:
        actual = getattr(multi_layer_encoder, name)
        desired = module
        assert actual is desired

    for name in tuple(zip(*modules))[0][idx + 1 :]:
        with pytest.raises(AttributeError):
            getattr(multi_layer_encoder, name)


def test_MultiLayerEncoder_trim_layers():
    layers = [str(idx) for idx in range(3)]
    modules = [(layer, nn.Module()) for layer in layers]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    for name, module in modules:
        actual = getattr(multi_layer_encoder, name)
        desired = module
        assert actual is desired

    idx = 1
    for layer in [str(idx) for idx in range(idx + 1)]:
        multi_layer_encoder.register_layer(layer)
    multi_layer_encoder.trim()

    for name, module in modules[: idx + 1]:
        actual = getattr(multi_layer_encoder, name)
        desired = module
        assert actual is desired

    for name in tuple(zip(*modules))[0][idx + 1 :]:
        with pytest.raises(AttributeError):
            getattr(multi_layer_encoder, name)


def test_MultiLayerEncoder_extract_encoder():
    conv = nn.Conv2d(3, 1, 1)
    relu = nn.ReLU(inplace=False)

    modules = (("conv", conv), ("relu", relu))
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    layer = "relu"
    single_layer_encoder = multi_layer_encoder.extract_encoder(layer)

    assert isinstance(single_layer_encoder, enc.SingleLayerEncoder)
    assert single_layer_encoder.multi_layer_encoder is multi_layer_encoder
    assert single_layer_encoder.layer == layer
    assert layer in multi_layer_encoder.registered_layers


def test_SingleLayerEncoder_call():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 1, 1)
    relu = nn.ReLU(inplace=False)
    input = torch.rand(1, 3, 128, 128)

    modules = (("conv", conv), ("relu", relu))
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    single_layer_encoder = enc.SingleLayerEncoder(multi_layer_encoder, "conv")

    actual = single_layer_encoder(input)
    desired = conv(input)
    ptu.assert_allclose(actual, desired)
