import itertools
from os import path

import pytest

from torchvision.datasets.utils import check_integrity

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
                    self.assertTrue(
                        check_integrity(
                            path.join(root, image_or_guide.file), md5=image_or_guide.md5
                        )
                    )
