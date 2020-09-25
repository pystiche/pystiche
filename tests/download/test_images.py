import itertools
from os import path

import pytest

from pystiche.demo import demo_images

from .utils import (
    assert_downloads_correctly,
    assert_is_downloadable,
    rate_limited_urlopen,
    retry,
)
from tests.mocks import make_mock_target

IMAGES_AND_IDS = [
    (image, f"demo, {name}") for name, image in itertools.chain(demo_images(),)
]

IMAGE_PARAMETRIZE_KWARGS = dict(zip(("argvalues", "ids"), zip(*IMAGES_AND_IDS)))


def assert_image_is_downloadable(image, **kwargs):
    assert_is_downloadable(image.url, **kwargs)


@pytest.mark.slow
@pytest.mark.parametrize("image", **IMAGE_PARAMETRIZE_KWARGS)
def test_image_download_smoke(subtests, image):
    retry(lambda: assert_image_is_downloadable(image), times=2, wait=5.0)


def assert_image_downloads_correctly(image, **kwargs):
    def downloader(url, root):
        image.download(root=root)
        return path.join(root, image.file)

    assert_downloads_correctly(None, image.md5, downloader=downloader, **kwargs)


@pytest.mark.large_download
@pytest.mark.slow
@pytest.mark.parametrize("image", **IMAGE_PARAMETRIZE_KWARGS)
def test_image_download(mocker, image):
    mocker.patch(make_mock_target("misc", "urlopen"), wraps=rate_limited_urlopen)
    assert_image_downloads_correctly(image)
