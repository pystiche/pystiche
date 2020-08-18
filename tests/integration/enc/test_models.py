import itertools

import pytest

import torch

import pystiche
from pystiche import enc

from tests import mocks


@pytest.fixture
def patch_multi_layer_encoder_load_weights(mocker):
    return mocks.patch_multi_layer_encoder_load_weights(mocker=mocker)


@pytest.mark.large_download
@pytest.mark.slow
@pytest.mark.flaky
def test_AlexNetMultiLayerEncoder(enc_asset_loader):
    asset = enc_asset_loader("alexnet")

    multi_layer_encoder = enc.alexnet_multi_layer_encoder(
        weights="torch", preprocessing=False, allow_inplace=False
    )
    layers = tuple(multi_layer_encoder.children_names())
    with torch.no_grad():
        encs = multi_layer_encoder(asset.input.image, layers)

    actual = dict(
        zip(
            layers,
            [pystiche.TensorKey(x, precision=asset.params.precision) for x in encs],
        )
    )
    desired = asset.output.enc_keys
    assert actual == desired


@pytest.mark.slow
def test_alexnet_multi_layer_encoder_smoke(patch_multi_layer_encoder_load_weights):
    multi_layer_encoder = enc.alexnet_multi_layer_encoder()
    assert isinstance(multi_layer_encoder, enc.alexnet.AlexNetMultiLayerEncoder)


@pytest.mark.slow
def test_AlexNetMultiLayerEncoder_repr_smoke(patch_multi_layer_encoder_load_weights):
    multi_layer_encoder = enc.alexnet_multi_layer_encoder()
    assert isinstance(repr(multi_layer_encoder), str)


@pytest.fixture(scope="module")
def vgg_archs():
    return tuple(
        f"vgg{num_layers}{'_bn' if batch_norm else ''}"
        for num_layers, batch_norm in itertools.product((11, 13, 16, 19), (False, True))
    )


@pytest.fixture(scope="module")
def vgg_multi_layer_encoder_loaders(vgg_archs):
    return tuple(getattr(enc, f"{arch}_multi_layer_encoder") for arch in vgg_archs)


@pytest.mark.large_download
@pytest.mark.slow
@pytest.mark.flaky
def test_VGGMultiLayerEncoder(
    subtests, vgg_archs, vgg_multi_layer_encoder_loaders, enc_asset_loader
):
    for arch, loader in zip(vgg_archs, vgg_multi_layer_encoder_loaders):
        with subtests.test(arch=arch):
            asset = enc_asset_loader(arch)

            multi_layer_encoder = loader(
                weights="torch", preprocessing=False, allow_inplace=False
            )
            layers = tuple(multi_layer_encoder.children_names())
            with torch.no_grad():
                encs = multi_layer_encoder(asset.input.image, layers)

            actual = dict(
                zip(
                    layers,
                    [
                        pystiche.TensorKey(x, precision=asset.params.precision)
                        for x in encs
                    ],
                )
            )
            desired = asset.output.enc_keys
            assert actual == desired


@pytest.mark.slow
def test_VGGMultiLayerEncoder_repr_smoke(
    subtests, vgg_multi_layer_encoder_loaders, patch_multi_layer_encoder_load_weights
):
    for loader in vgg_multi_layer_encoder_loaders:
        with subtests.test(fn=loader.__name__):
            multi_layer_encoder = loader()
            assert isinstance(repr(multi_layer_encoder), str)


@pytest.mark.slow
def test_vgg_multi_layer_encoder_smoke(
    subtests, vgg_multi_layer_encoder_loaders, patch_multi_layer_encoder_load_weights
):
    for loader in vgg_multi_layer_encoder_loaders:
        with subtests.test(fn=loader.__name__):
            multi_layer_encoder = loader()
            assert isinstance(multi_layer_encoder, enc.vgg.VGGMultiLayerEncoder)
