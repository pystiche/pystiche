import pytest

from torch import hub
from torchvision import models

from tests.mocks import ContextMock


@pytest.fixture
def main_archs():
    return ("vgg16", "vgg19")


@pytest.fixture
def side_archs():
    return ("vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16_bn", "vgg19_bn")


@pytest.fixture
def frameworks():
    return ("caffe", "torch")


@pytest.fixture
def patch_vgg_load_state_dict_from_url(mocker):
    def patch_load_state_dict_from_url_(arch):
        model = getattr(models, arch)(pretrained=False)
        mock = mocker.patch(
            "torch.hub.load_state_dict_from_url", return_value=model.state_dict(),
        )
        return ContextMock(mock)

    return patch_load_state_dict_from_url_


@pytest.mark.slow
def test_vgg_main_archs(subtests, github, main_archs, frameworks):
    for arch, framework in zip(main_archs, frameworks):
        with subtests.test(arch=arch, framework=framework):
            assert hub.load(github, arch, pretrained=True, framework=framework)


@pytest.mark.slow
def test_vgg_side_archs_torch(
    subtests, patch_vgg_load_state_dict_from_url, github, side_archs
):
    for arch in side_archs:
        with subtests.test(arch=arch), patch_vgg_load_state_dict_from_url(arch):
            assert hub.load(github, arch, pretrained=True, framework="torch")


@pytest.mark.slow
def test_vgg_side_archs_caffe(subtests, github, side_archs):
    for arch in side_archs:
        with subtests.test(arch=arch):
            with pytest.raises(RuntimeError):
                assert hub.load(github, arch, pretrained=True, framework="caffe")
