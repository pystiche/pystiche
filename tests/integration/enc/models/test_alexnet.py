import pytest

import torch
from torchvision import models

import pystiche
from pystiche import enc


def test_alexnet_multi_layer_encoder_smoke(subtests):
    multi_layer_encoder = enc.alexnet_multi_layer_encoder(pretrained=False)
    assert isinstance(multi_layer_encoder, enc.alexnet.AlexNetMultiLayerEncoder)

    with subtests.test("repr"):
        assert isinstance(multi_layer_encoder, enc.alexnet.AlexNetMultiLayerEncoder)


@pytest.mark.large_download
@pytest.mark.slow
@pytest.mark.flaky
def test_alexnet_multi_layer_encoder(enc_asset_loader):
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


def test_alexnet_multi_layer_encoder_state_dict_url(subtests, frameworks):
    def should_be_available(framework):
        return framework == "torch"

    multi_layer_encoder = enc.alexnet_multi_layer_encoder(pretrained=False)

    for framework in frameworks:
        with subtests.test(framework=framework):
            if should_be_available(framework):
                assert isinstance(multi_layer_encoder.state_dict_url(framework), str)
            else:
                with pytest.raises(RuntimeError):
                    multi_layer_encoder.state_dict_url(framework)


@pytest.mark.slow
def test_alexnet_multi_layer_encoder_load_state_dict_smoke():
    model = models.alexnet(pretrained=False)
    state_dict = model.state_dict()

    multi_layer_encoder = enc.alexnet_multi_layer_encoder()
    multi_layer_encoder.load_state_dict(state_dict)
