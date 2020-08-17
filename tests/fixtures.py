import functools

import pytest

from torch import nn

from . import assets, utils


@pytest.fixture(scope="session", autouse=True)
def watch_project_dir():
    with utils.watch_dir("."):
        yield


@pytest.fixture(scope="session")
def test_image_file():
    return assets.get_image_file("test_image")


@pytest.fixture(scope="session")
def test_image_url():
    return assets.get_image_url("test_image")


@pytest.fixture
def test_image():
    return assets.read_image("test_image")


@pytest.fixture
def test_image_pil():
    return assets.read_image("test_image", pil=True)


@pytest.fixture(scope="session")
def enc_asset_loader():
    return functools.partial(assets.load_asset, "enc")


@pytest.fixture(scope="session")
def optim_asset_loader():
    return functools.partial(assets.load_asset, "optim")


@pytest.fixture
def forward_pass_counter():
    class ForwardPassCounter(nn.Module):
        def __init__(self):
            super().__init__()
            self.count = 0

        def forward(self, input):
            self.count += 1
            return input

    return ForwardPassCounter()
