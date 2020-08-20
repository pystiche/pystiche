import pytest


@pytest.fixture(scope="package")
def frameworks():
    return ("torch", "caffe")
