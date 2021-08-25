import itertools

import pytest
import pytorch_testing_utils as ptu

import torch

from pystiche import enc

from tests import mocks

FRAMEWORKS = ("torch", "caffe")


def _get_processing_cls(framework, position):
    return getattr(enc, f"{framework.title()}{position.title()}processing")


class TestPreProcessing:
    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_main(self, framework):
        assert isinstance(
            enc.preprocessing(framework), _get_processing_cls(framework, "pre")
        )


    def test_unknown_framework(self):
        with pytest.raises(ValueError):
            enc.preprocessing("unknown")


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_get_preprocessor_deprecation(mocker, framework):
    mock = mocker.patch(
        mocks.make_mock_target("enc", "prepostprocessing", "preprocessing")
    )
    with pytest.warns(UserWarning):
        enc.get_preprocessor(framework)

    mock.assert_called_with(framework)


class TestPostProcessing:
    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_main(self, framework):
        assert isinstance(
            enc.postprocessing(framework), _get_processing_cls(framework, "post")
        )


    def test_unknown_framework(self):
        with pytest.raises(ValueError):
            enc.postprocessing("unknown")


def collect_processings(*argnames):
    values = [
        _get_processing_cls(framework, position)()
        for framework, position in itertools.product(FRAMEWORKS, ("pre", "post"))
    ]
    return pytest.mark.parametrize(
        argnames, [pytest.param(value, id=type(value).__name__) for value in values],
    )


def collect_processing_pairs(*argnames):
    return pytest.mark.parametrize(
        argnames,
        [
            pytest.param(
                _get_processing_cls(framework, "pre")(),
                _get_processing_cls(framework, "post")(),
                id=framework,
            )
            for framework in FRAMEWORKS
        ],
    )


@collect_processings("processing")
def test_repr_smoke(processing):
    assert isinstance(repr(processing), str)


def _make_pixel(values):
    return torch.tensor(values).view(1, -1, 1, 1)


@pytest.fixture
def pixel():
    return _make_pixel((0.0, 0.5, 1.0))


@collect_processing_pairs("preprocessing", "postprocessing")
def test_cycle_consistency(pixel, preprocessing, postprocessing):
    ptu.assert_allclose(postprocessing(preprocessing(pixel)), pixel, rtol=1e-6)
    ptu.assert_allclose(preprocessing(postprocessing(pixel)), pixel, rtol=1e-6)


@pytest.fixture
def torch_mean():
    return _make_pixel((0.485, 0.456, 0.406))


@pytest.fixture
def torch_std():
    return _make_pixel((0.229, 0.224, 0.225))


def test_TorchPreprocessing(pixel, torch_mean, torch_std):
    processing = enc.TorchPreprocessing()

    expected = pixel
    input = pixel.mul(torch_std).add(torch_mean)
    actual = processing(input)

    ptu.assert_allclose(actual, expected, rtol=1e-6)


def test_TorchPostprocessing(pixel, torch_mean, torch_std):
    processing = enc.TorchPostprocessing()

    expected = pixel
    input = pixel.sub(torch_mean).div(torch_std)
    actual = processing(input)

    ptu.assert_allclose(actual, expected, rtol=1e-6)


@pytest.fixture
def caffe_mean():
    return _make_pixel((0.485, 0.458, 0.408))


@pytest.fixture
def caffe_std():
    return _make_pixel((1.0, 1.0, 1.0))


def test_CaffePreprocessing(pixel, caffe_mean, caffe_std):
    processing = enc.CaffePreprocessing()

    expected = pixel
    input = pixel.flip(1).div(255).mul(caffe_std).add(caffe_mean)
    actual = processing(input)

    ptu.assert_allclose(actual, expected, rtol=1e-6)


def test_CaffePostprocessing(pixel, caffe_mean, caffe_std):
    processing = enc.CaffePostprocessing()

    expected = pixel
    input = pixel.sub(caffe_mean).div(caffe_std).mul(255).flip(1)
    actual = processing(input)

    ptu.assert_allclose(actual, expected, rtol=1e-6)
