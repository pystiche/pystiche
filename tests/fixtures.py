import pytest

from . import assets


@pytest.fixture(scope="session")
def test_image_file():
    return assets.get_image_file("test_image")


@pytest.fixture(scope="session")
def test_image_url():
    return assets.get_image_url("test_image")


@pytest.fixture
def test_image():
    return assets.read_image("test_image")
