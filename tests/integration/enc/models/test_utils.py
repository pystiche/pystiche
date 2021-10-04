import functools
from collections import OrderedDict
from copy import copy
from urllib.parse import urljoin

import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

from pystiche import enc

from tests import asserts, mocks


class TestSelectURL:
    def test_main(self):
        key = "available"
        url = "https://pystiche.org"

        assert enc.select_url({key: url}, key) == url

    def test_not_available(self):
        key = "not_available"
        urls = {}

        with pytest.raises(RuntimeError):
            enc.select_url(urls, key)

    def test_custom_format(self):
        key = "not_available"
        urls = {}

        def format(key):
            return f"custom format for '{key}'"

        expected = format(key)

        with pytest.raises(RuntimeError) as exc_info:
            enc.select_url(urls, key, format=format)

            actual = exc_info.value.args[0]
            assert actual.endswith(expected)


@pytest.fixture
def multi_layer_encoder_urls(frameworks):
    base = "https://download.pystiche.org/models/"

    return {
        framework: urljoin(base, f"{framework}-01234567.pth")
        for framework in frameworks
    }


@pytest.fixture
def multi_layer_encoder_modules():
    return [("conv", nn.Conv2d(3, 3, 1)), ("relu", nn.ReLU())]


@pytest.fixture
def multi_layer_encoder_state_dict(multi_layer_encoder_modules):
    return OrderedDict(
        (f"{name}.{key}", tensor)
        for name, module in multi_layer_encoder_modules
        for key, tensor in module.state_dict().items()
    )


@pytest.fixture
def other_multi_layer_encoder_state_dict(multi_layer_encoder_state_dict):
    return OrderedDict(
        (id, torch.zeros_like(tensor))
        for id, tensor in multi_layer_encoder_state_dict.items()
    )


@pytest.fixture
def multi_layer_encoder_state_dict_key_map(multi_layer_encoder_modules):
    return {
        f"{idx}.{key}": f"{name}.{key}"
        for idx, (name, module) in enumerate(multi_layer_encoder_modules)
        for key in module.state_dict().keys()
    }


@pytest.fixture
def multi_layer_encoder_cls(
    mocker,
    multi_layer_encoder_urls,
    multi_layer_encoder_modules,
    multi_layer_encoder_state_dict_key_map,
):
    class MockModelMultiLayerEncoder(enc.ModelMultiLayerEncoder):
        def state_dict_url(self, framework: str) -> str:
            return enc.select_url(multi_layer_encoder_urls, framework)

        def collect_modules(self, inplace: bool):
            return (
                copy(multi_layer_encoder_modules),
                multi_layer_encoder_state_dict_key_map,
            )

    mocker.patch(
        mocks.make_mock_target(
            "enc", "models", "utils", "hub", "load_state_dict_from_url"
        ),
        return_value=MockModelMultiLayerEncoder(pretrained=False).state_dict(),
    )

    return MockModelMultiLayerEncoder


class TestModelMultiLayerEncoder:
    def test_internal_preprocessing(
        self, frameworks, multi_layer_encoder_cls
    ):
        for framework in ("caffe",):
            multi_layer_encoder = multi_layer_encoder_cls(
                pretrained=False, framework=framework, internal_preprocessing=True
            )
            assert "preprocessing" in multi_layer_encoder
            assert isinstance(
                multi_layer_encoder.preprocessing,
                type(enc.get_preprocessor(framework)),
            )

    def test_pretrained(self, mocker, multi_layer_encoder_cls):
        load_state_dict_from_url = mocker.patch(
            mocks.make_mock_target(
                "enc",
                "models",
                "utils",
                "ModelMultiLayerEncoder",
                "load_state_dict_from_url",
            )
        )

        framework = "framework"
        multi_layer_encoder_cls(
            pretrained=True, framework=framework, internal_preprocessing=False
        )

        load_state_dict_from_url.assert_called_once_with(framework)

    def test_load_state_dict(
        self, multi_layer_encoder_cls, other_multi_layer_encoder_state_dict,
    ):
        multi_layer_encoder = multi_layer_encoder_cls(pretrained=False)
        multi_layer_encoder.load_state_dict(
            other_multi_layer_encoder_state_dict, map_names=False
        )

        ptu.assert_allclose(
            multi_layer_encoder.state_dict(), other_multi_layer_encoder_state_dict
        )

    def test_load_state_dict_name_mapping(
        self,
        multi_layer_encoder_cls,
        multi_layer_encoder_state_dict,
        other_multi_layer_encoder_state_dict,
        multi_layer_encoder_state_dict_key_map,
    ):
        inv_key_map = {v: k for k, v in multi_layer_encoder_state_dict_key_map.items()}
        inv_state_dict = OrderedDict(
            (inv_key_map[name], tensor)
            for name, tensor in other_multi_layer_encoder_state_dict.items()
        )

        multi_layer_encoder = multi_layer_encoder_cls(pretrained=False)
        multi_layer_encoder.load_state_dict(inv_state_dict, map_names=True)

        ptu.assert_allclose(
            multi_layer_encoder.state_dict(), other_multi_layer_encoder_state_dict
        )

    def test_repr(self, mocker, multi_layer_encoder_cls):
        mocker.patch(
            mocks.make_mock_target(
                "enc",
                "models",
                "utils",
                "ModelMultiLayerEncoder",
                "load_state_dict_from_url",
            )
        )

        framework = "framework"
        internal_preprocessing = False
        allow_inplace = True
        cls = functools.partial(
            multi_layer_encoder_cls,
            framework=framework,
            internal_preprocessing=internal_preprocessing,
            allow_inplace=allow_inplace,
        )

        for pretrained in (True, False):
            multi_layer_encoder = cls(pretrained=pretrained)
            assert_property_in_repr_ = functools.partial(
                asserts.assert_property_in_repr, repr(multi_layer_encoder)
            )

            if pretrained:
                assert_property_in_repr_("framework", framework)
            else:
                assert_property_in_repr_("pretrained", False)

            assert_property_in_repr_(
                "internal_preprocessing", internal_preprocessing
            )

            assert_property_in_repr_("allow_inplace", allow_inplace)
