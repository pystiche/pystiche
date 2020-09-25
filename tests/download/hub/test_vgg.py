import itertools

import pytest

from torch import hub
from torchvision import models

MAIN_ARCHS = ("vgg16", "vgg19")
SIDE_ARCHS = (
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
)
FRAMEWORKS = ("caffe", "torch")


@pytest.mark.large_download
@pytest.mark.slow
@pytest.mark.parametrize(
    ("arch", "framework"), list(itertools.product(MAIN_ARCHS, FRAMEWORKS))
)
def test_vgg_main_archs(github, arch, framework):
    assert hub.load(github, arch, pretrained=True, framework=framework)


@pytest.mark.slow
@pytest.mark.parametrize("arch", SIDE_ARCHS)
def test_vgg_side_archs_torch(mocker, github, arch):
    model = getattr(models, arch)(pretrained=False)
    mocker.patch(
        "torch.hub.load_state_dict_from_url", return_value=model.state_dict(),
    )

    assert hub.load(github, arch, pretrained=True, framework="torch")


@pytest.mark.slow
@pytest.mark.parametrize("arch", SIDE_ARCHS)
def test_vgg_side_archs_caffe(subtests, github, arch):
    with pytest.raises(RuntimeError):
        hub.load(github, arch, pretrained=True, framework="caffe")
