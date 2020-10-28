import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

from pystiche import enc

from tests.asserts import assert_named_modules_identical


def test_MultiLayerEncoder():
    modules = [(str(idx), nn.Module()) for idx in range(3)]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    for name, module in modules:
        actual = getattr(multi_layer_encoder, name)
        desired = module
        assert actual is desired


def test_MultiLayerEncoder_named_children():
    modules = [(str(idx), nn.Module()) for idx in range(3)]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    actual = tuple(multi_layer_encoder.children_names())
    desired = tuple(zip(*modules))[0]
    assert actual == desired


def test_MultiLayerEncoder_contains():
    idcs = (0, 2)
    modules = [(str(idx), nn.Module()) for idx in idcs]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    for idx in idcs:
        assert str(idx) in multi_layer_encoder

    for idx in set(range(max(idcs))) - set(idcs):
        assert str(idx) not in multi_layer_encoder


def test_MultiLayerEncoder_extract_deepest_layer():
    layers = [str(idx) for idx in range(3)]
    modules = [(layer, nn.Module()) for layer in layers]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    actual = multi_layer_encoder.extract_deepest_layer(layers)
    desired = layers[-1]
    assert actual == desired

    actual = multi_layer_encoder.extract_deepest_layer(sorted(layers, reverse=True))
    desired = layers[-1]
    assert actual == desired

    del multi_layer_encoder._modules[layers[-1]]

    with pytest.raises(ValueError):
        multi_layer_encoder.extract_deepest_layer(layers)

    layers = layers[:-1]

    actual = multi_layer_encoder.extract_deepest_layer(layers)
    desired = layers[-1]
    assert actual == desired


def test_MultiLayerEncoder_named_children_to():
    layers = [str(idx) for idx in range(3)]
    modules = [(layer, nn.Module()) for layer in layers]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    actuals = multi_layer_encoder.named_children_to(layers[-2])
    desireds = modules[:-2]
    assert_named_modules_identical(actuals, desireds)

    actuals = multi_layer_encoder.named_children_to(layers[-2], include_last=True)
    desireds = modules[:-1]
    assert_named_modules_identical(actuals, desireds)


def test_MultiLayerEncoder_named_children_from():
    layers = [str(idx) for idx in range(3)]
    modules = [(layer, nn.Module()) for layer in layers]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    actuals = multi_layer_encoder.named_children_from(layers[-2])
    desireds = modules[1:]
    assert_named_modules_identical(actuals, desireds)

    actuals = multi_layer_encoder.named_children_from(layers[-2], include_first=False)
    desireds = modules[2:]
    assert_named_modules_identical(actuals, desireds)


def test_MultiLayerEncoder_call():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 1, 1)
    relu = nn.ReLU(inplace=False)
    pool = nn.MaxPool2d(2)
    input = torch.rand(1, 3, 128, 128)

    modules = (("conv", conv), ("relu", relu), ("pool", pool))
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    layers = ("conv", "pool")
    encs = multi_layer_encoder(input, layers)

    actual = encs[0]
    desired = conv(input)
    ptu.assert_allclose(actual, desired)

    actual = encs[1]
    desired = pool(relu(conv(input)))
    ptu.assert_allclose(actual, desired)


def test_MultiLayerEncoder_call_store(forward_pass_counter):
    torch.manual_seed(0)

    modules = (("count", forward_pass_counter),)
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    layers = ("count",)
    input = torch.rand(1, 3, 128, 128)
    multi_layer_encoder(input, layers, store=True)
    multi_layer_encoder(input, layers)

    actual = forward_pass_counter.count
    desired = 1
    assert actual == desired

    new_input = torch.rand(1, 3, 128, 128)
    multi_layer_encoder(new_input, layers)

    actual = forward_pass_counter.count
    desired = 2
    assert actual == desired


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


def test_MultiLayerEncoder_encode(forward_pass_counter):
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 1, 1)
    relu = nn.ReLU(inplace=False)
    input = torch.rand(1, 3, 128, 128)

    modules = (("count", forward_pass_counter), ("conv", conv), ("relu", relu))
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    layers = ("conv", "relu")
    multi_layer_encoder.registered_layers.update(layers)
    multi_layer_encoder.encode(input)
    encs = multi_layer_encoder(input, layers)

    actual = encs[0]
    desired = conv(input)
    ptu.assert_allclose(actual, desired)

    actual = encs[1]
    desired = relu(conv(input))
    ptu.assert_allclose(actual, desired)

    actual = forward_pass_counter.count
    desired = 1
    assert actual == desired


def test_MultiLayerEncoder_empty_storage(forward_pass_counter):
    torch.manual_seed(0)
    input = torch.rand(1, 3, 128, 128)

    modules = (("count", forward_pass_counter),)
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    layers = ("count",)
    multi_layer_encoder(input, layers, store=True)
    multi_layer_encoder.empty_storage()
    multi_layer_encoder(input, layers)

    actual = forward_pass_counter.count
    desired = 2
    assert actual == desired


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
    multi_layer_encoder.registered_layers.update([str(idx) for idx in range(idx + 1)])
    multi_layer_encoder.trim()

    for name, module in modules[: idx + 1]:
        actual = getattr(multi_layer_encoder, name)
        desired = module
        assert actual is desired

    for name in tuple(zip(*modules))[0][idx + 1 :]:
        with pytest.raises(AttributeError):
            getattr(multi_layer_encoder, name)


def test_MultiLayerEncoder_call_future_warning():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 1, 1)
    input = torch.rand(1, 3, 1, 1)

    modules = (("conv", conv),)
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    with pytest.warns(FutureWarning):
        multi_layer_encoder(input, ("conv",))


def test_MultiLayerEncoder_encode_future_warning():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 1, 1)
    input = torch.rand(1, 3, 1, 1)

    modules = (("conv", conv),)
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    with pytest.warns(FutureWarning):
        multi_layer_encoder.encode(input)


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
