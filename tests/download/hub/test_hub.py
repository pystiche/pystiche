import pytest

from torch import hub, nn

ENTRY_POINTS = (
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
)


def test_hub_entrypoints(github):
    assert set(hub.list(github)) == set(ENTRY_POINTS)


@pytest.mark.parametrize("model", ENTRY_POINTS)
def test_hub_help_smoke(github, model):
    assert isinstance(hub.help(github, model), str)


@pytest.mark.slow
@pytest.mark.parametrize("model", ENTRY_POINTS)
def test_hub_load_smoke(github, model):
    assert isinstance(hub.load(github, model), nn.Module)
