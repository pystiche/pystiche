import itertools
import logging
from os import path

import pytest

from torchvision.datasets.utils import calculate_md5

from pystiche import demo


@pytest.mark.large_download
@pytest.mark.slow
def test_demo_images_smoke(subtests, tmpdir):
    images = demo.demo_images()
    images.download(root=tmpdir)
    images = [image for _, image in images]

    guides = [
        [guide for _, guide in image.guides]
        for image in images
        if image.guides is not None
    ]

    for image_or_guide in itertools.chain(images, *guides):
        with subtests.test(image_or_guide=image_or_guide):
            file = path.join(tmpdir, image_or_guide.file)

            assert path.exists(file), f"File {file} does not exist."

            actual = calculate_md5(file)
            desired = image_or_guide.md5
            assert actual == desired, (
                f"The actual and desired MD5 hash of the image mismatch: "
                f"{actual} != {desired}"
            )


def test_demo_logger_smoke(caplog):
    optim_logger = demo.demo_logger()

    with caplog.at_level(logging.INFO, optim_logger.logger.name):
        optim_logger.message("test message")

    assert caplog.records
