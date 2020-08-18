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


@pytest.mark.large_download
@pytest.mark.slow
@pytest.mark.flaky
def test_VGGMultiLayerEncoder(subtests, enc_asset_loader):
    archs = ("vgg11", "vgg13", "vgg16", "vgg19")
    archs = (*archs, *[f"{arch}_bn" for arch in archs])

    for arch in archs:
        with subtests.test(arch=arch):
            asset = enc_asset_loader(arch)

            get_vgg_multi_layer_encoder = enc.__dict__[f"{arch}_multi_layer_encoder"]
            multi_layer_encoder = get_vgg_multi_layer_encoder(
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
def test_vgg_multi_layer_encoder_smoke(
    subtests, patch_multi_layer_encoder_load_weights
):
    fns = (
        enc.vgg11_multi_layer_encoder,
        enc.vgg11_bn_multi_layer_encoder,
        enc.vgg13_multi_layer_encoder,
        enc.vgg13_bn_multi_layer_encoder,
        enc.vgg16_multi_layer_encoder,
        enc.vgg16_bn_multi_layer_encoder,
        enc.vgg19_multi_layer_encoder,
        enc.vgg19_bn_multi_layer_encoder,
    )
    for fn in fns:
        with subtests.test(fn=fn.__name__):
            multi_layer_encoder = fn()
            assert isinstance(multi_layer_encoder, enc.vgg.VGGMultiLayerEncoder)
