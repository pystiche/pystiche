import pytest

from torch import hub, nn

from tests import utils


@pytest.fixture(scope="module")
def entry_points():
    globals, _ = utils.exec_file("hubconf.py")
    return tuple(
        key for key, val in globals.items() if not key.startswith("_") and callable(val)
    )


def test_hub_entrypoints(github, entry_points):
    models = hub.list(github)
    assert set(models) == set(entry_points)


def test_hub_help_smoke(subtests, github, entry_points):
    for model in entry_points:
        with subtests.test(model):
            assert isinstance(hub.help(github, model), str)


@pytest.mark.slow
def test_hub_load_smoke(subtests, github, entry_points):
    for model in entry_points:
        with subtests.test(model):
            assert isinstance(hub.load(github, model), nn.Module)
