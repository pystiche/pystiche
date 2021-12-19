from os import path

import pytest
import pytorch_testing_utils as ptu
from PIL import Image

import torch

from pystiche import image


def get_test_rgb_triplet(channel):
    rgb_triplet = [0] * 3
    rgb_triplet[channel] = 255
    return tuple(rgb_triplet)


def get_test_guides(block_size=(10, 30)):
    def get_guide(channel):
        size = (1, 1, *block_size)
        blocks = [torch.zeros(size)] * 3
        blocks[channel] = torch.ones(size)
        return torch.cat(blocks, dim=2)

    guides = {}
    color_map = {}
    for channel, region in enumerate("RGB"):
        guides[region] = get_guide(channel)

        rgb_triplet = get_test_rgb_triplet(channel)
        color_map[region] = rgb_triplet

    return guides, color_map


def get_test_segmentation(block_size=(10, 30)):
    def get_colored_block(channel):
        block = torch.zeros(1, 3, *block_size)
        block[:, channel, :, :] = 1.0
        return block

    blocks = []
    region_map = {}
    for channel, region in enumerate("RGB"):
        blocks.append(get_colored_block(channel))

        rgb_triplet = get_test_rgb_triplet(channel)
        region_map[rgb_triplet] = region

    segmentation = torch.cat(blocks, dim=2)

    return segmentation, region_map


def write_guide(guide, file):
    guide = guide.squeeze().byte().mul(255).numpy()
    Image.fromarray(guide, mode="L").save(file)


class TestVerifyGuides:
    def test_main(self):
        guides, _ = get_test_guides()
        image.verify_guides(guides)

    def test_coverage(self):
        guides, _ = get_test_guides()
        del guides["R"]

        with pytest.raises(RuntimeError):
            image.verify_guides(guides)

        image.verify_guides(guides, verify_coverage=False)

    def test_overlap(self):
        guides, _ = get_test_guides()
        guides["R2"] = guides["R"]

        with pytest.raises(RuntimeError):
            image.verify_guides(guides)

        image.verify_guides(guides, verify_overlap=False)


def test_read_guides(tmpdir):
    guides, _ = get_test_guides()

    for region, guide in guides.items():
        write_guide(guide, path.join(tmpdir, f"{region}.png"))

    actual = image.read_guides(tmpdir)
    desired = guides
    ptu.assert_allclose(actual, desired)


def test_write_guides(tmpdir):
    guides, _ = get_test_guides()

    image.write_guides(guides, tmpdir)

    actual = {
        region: image.read_image(file=path.join(tmpdir, f"{region}.png"), mode="L")
        for region in guides.keys()
    }
    desired = guides

    ptu.assert_allclose(actual, desired)


def test_guides_to_segmentation():
    guides, color_map = get_test_guides()
    segmentation, _ = get_test_segmentation()

    actual = image.guides_to_segmentation(guides, color_map=color_map)
    desired = segmentation
    ptu.assert_allclose(actual, desired)


def test_segmentation_to_guides():
    guides, _ = get_test_guides()
    segmentation, region_map = get_test_segmentation()

    actual = image.segmentation_to_guides(segmentation, region_map=region_map)
    desired = guides
    ptu.assert_allclose(actual, desired)
