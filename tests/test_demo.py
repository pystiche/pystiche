import itertools
from os import path

import pytest

from torchvision.datasets.utils import calculate_md5

from pystiche import demo

from .utils import PysticheTestCase, get_tmp_dir


class TestDemo(PysticheTestCase):
    @pytest.mark.large_download
    @pytest.mark.slow
    def test_demo_images_smoke(self):

        with get_tmp_dir() as root:
            images = demo.demo_images()
            images.download(root=root)
            images = [image for _, image in images]

            guides = [
                [guide for _, guide in image.guides]
                for image in images
                if image.guides is not None
            ]

            for image_or_guide in itertools.chain(images, *guides):
                with self.subTest(image_or_guide=image_or_guide):
                    file = path.join(root, image_or_guide.file)

                    self.assertTrue(
                        path.exists(file), msg=f"File {file} does not exist."
                    )

                    actual = calculate_md5(file)
                    desired = image_or_guide.md5
                    self.assertEqual(
                        actual,
                        desired,
                        msg=(
                            f"The actual and desired MD5 hash of the image mismatch: "
                            f"{actual} != {desired}"
                        ),
                    )

    def test_demo_logger_smoke(self):
        optim_logger = demo.demo_logger()

        with self.assertLogs(optim_logger.logger, "INFO"):
            optim_logger.message("Test message")
