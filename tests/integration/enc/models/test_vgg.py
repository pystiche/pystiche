import itertools

import pytest
import pytorch_testing_utils as ptu

import torch

import pystiche
from pystiche import enc
from pystiche.enc.models.vgg import MODELS

from tests import asserts, mocks


@pytest.fixture(scope="module")
def vgg_archs():
    return tuple(
        f"vgg{num_layers}{'_bn' if batch_norm else ''}"
        for num_layers, batch_norm in itertools.product((11, 13, 16, 19), (False, True))
    )


@pytest.fixture(scope="module")
def vgg_multi_layer_encoder_loaders(vgg_archs):
    return tuple(getattr(enc, f"{arch}_multi_layer_encoder") for arch in vgg_archs)


# We use VGG11 with batch normalization as proxy for all VGG architectures
vgg_multi_layer_encoder = enc.vgg11_bn_multi_layer_encoder
vgg = MODELS["vgg11_bn"]


class TestVGG:
    @pytest.mark.slow
    def test_pretrained(self, mocker):
        state_dict = vgg(pretrained=False).state_dict()
        mocker.patch(
            mocks.make_mock_target(
                "enc", "models", "vgg", "hub", "load_state_dict_from_url"
            ),
            return_value=state_dict,
        )

        model = vgg(pretrained=True)
        ptu.assert_allclose(model.state_dict(), state_dict)

    def test_pretrained_num_classes_mismatch(self):
        with pytest.raises(RuntimeError):
            vgg(pretrained=True, num_classes=1001)


class TestVGGMultiLayerEncoder:
    @pytest.mark.slow
    def test_unknown_arch(self):
        arch = "unknown"
        with pytest.raises(ValueError):
            enc.VGGMultiLayerEncoder(
                arch, pretrained=False, internal_preprocessing=False
            )

    @pytest.mark.slow
    def test_smoke(
        self, vgg_archs, vgg_multi_layer_encoder_loaders,
    ):
        for arch, loader in zip(vgg_archs, vgg_multi_layer_encoder_loaders):
            multi_layer_encoder = loader(pretrained=False)
            assert isinstance(multi_layer_encoder, enc.vgg.VGGMultiLayerEncoder)
            asserts.assert_property_in_repr(
                repr(multi_layer_encoder), "arch", arch
            )

    @pytest.mark.large_download
    @pytest.mark.slow
    @pytest.mark.flaky
    def test_main(
        self, vgg_archs, vgg_multi_layer_encoder_loaders, enc_asset_loader
    ):
        for arch, loader in zip(vgg_archs, vgg_multi_layer_encoder_loaders):
            asset = enc_asset_loader(arch)
            multi_layer_encoder = loader(
                pretrained=True,
                weights="torch",
                preprocessing=False,
                allow_inplace=False,
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
    def test_state_dict_url(self, vgg_archs, frameworks):
        def should_be_available(arch, framework):
            if framework == "caffe" and arch in ("vgg16", "vgg19"):
                return True

            return framework == "torch"

        multi_layer_encoder = vgg_multi_layer_encoder(pretrained=False)

        for arch, framework in zip(vgg_archs, frameworks):
            if should_be_available(arch, framework):
                assert isinstance(
                    multi_layer_encoder.state_dict_url(framework), str
                )
            else:
                with pytest.raises(RuntimeError):
                    multi_layer_encoder.state_dict_url(framework)

    @pytest.mark.slow
    def test_load_state_dict_smoke(self):
        model = vgg(pretrained=False)
        state_dict = model.state_dict()

        multi_layer_encoder = vgg_multi_layer_encoder()
        multi_layer_encoder.load_state_dict(state_dict)
