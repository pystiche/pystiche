import pytest

import torch

import pystiche
from pystiche import enc


@pytest.mark.large_download
@pytest.mark.slow
@pytest.mark.flaky
def test_AlexNetMultiLayerEncoder(enc_asset_loader):
    asset = enc_asset_loader("alexnet")

    multi_layer_encoder = enc.alexnet_multi_layer_encoder(
        pretrained=True, weights="torch", preprocessing=False, allow_inplace=False
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


def test_alexnet_multi_layer_encoder_smoke(subtests):
    multi_layer_encoder = enc.alexnet_multi_layer_encoder(pretrained=False)
    assert isinstance(multi_layer_encoder, enc.alexnet.AlexNetMultiLayerEncoder)

    with subtests.test("repr"):
        assert isinstance(multi_layer_encoder, enc.alexnet.AlexNetMultiLayerEncoder)
