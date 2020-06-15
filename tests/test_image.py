from os import path
from unittest import mock

from PIL import Image

import torch

from pystiche.image import guides as guides_
from pystiche.image import io, utils

from .utils import PysticheTestCase, get_tmp_dir


class TestGuides(PysticheTestCase):
    @staticmethod
    def get_test_rgb_triplet(channel):
        rgb_triplet = [0] * 3
        rgb_triplet[channel] = 255
        return tuple(rgb_triplet)

    @staticmethod
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

            rgb_triplet = TestGuides.get_test_rgb_triplet(channel)
            color_map[region] = rgb_triplet

        return guides, color_map

    @staticmethod
    def get_test_segmentation(block_size=(10, 30)):
        def get_colored_block(channel):
            block = torch.zeros(1, 3, *block_size)
            block[:, channel, :, :] = 1.0
            return block

        blocks = []
        region_map = {}
        for channel, region in enumerate("RGB"):
            blocks.append(get_colored_block(channel))

            rgb_triplet = TestGuides.get_test_rgb_triplet(channel)
            region_map[rgb_triplet] = region

        segmentation = torch.cat(blocks, dim=2)

        return segmentation, region_map

    @staticmethod
    def write_guide(guide, file):
        guide = guide.squeeze().byte().mul(255).numpy()
        Image.fromarray(guide, mode="L").save(file)

    def test_verify_guides(self):
        guides, _ = self.get_test_guides()
        guides_.verify_guides(guides)

    def test_verify_guides_coverage(self):
        guides, _ = self.get_test_guides()
        del guides["R"]

        with self.assertRaises(RuntimeError):
            guides_.verify_guides(guides)

        guides_.verify_guides(guides, verify_coverage=False)

    def test_verify_guides_overlap(self):
        guides, _ = self.get_test_guides()
        guides["R2"] = guides["R"]

        with self.assertRaises(RuntimeError):
            guides_.verify_guides(guides)

        guides_.verify_guides(guides, verify_overlap=False)

    def test_read_guides(self):
        guides, _ = self.get_test_guides()

        with get_tmp_dir() as tmp_dir:
            for region, guide in guides.items():
                self.write_guide(guide, path.join(tmp_dir, f"{region}.png"))

            actual = guides_.read_guides(tmp_dir)
            desired = guides
            self.assertTensorDictAlmostEqual(actual, desired)

    def test_write_guides(self):
        guides, _ = self.get_test_guides()

        with get_tmp_dir() as tmp_dir:
            guides_.write_guides(guides, tmp_dir)

            actual = {
                region: self.load_image(file=path.join(tmp_dir, f"{region}.png"))
                for region in guides.keys()
            }
            desired = guides

            self.assertTensorDictAlmostEqual(actual, desired)

    def test_guides_to_segmentation(self):
        guides, color_map = self.get_test_guides()
        segmentation, _ = self.get_test_segmentation()

        actual = guides_.guides_to_segmentation(guides, color_map=color_map)
        desired = segmentation
        self.assertTensorAlmostEqual(actual, desired)

    def test_segmentation_to_guides(self):
        guides, _ = self.get_test_guides()
        segmentation, region_map = self.get_test_segmentation()

        actual = guides_.segmentation_to_guides(segmentation, region_map=region_map)
        desired = guides
        self.assertTensorDictAlmostEqual(actual, desired)


class TestIo(PysticheTestCase):
    def test_read_image(self):
        actual = io.read_image(self.default_image_file())
        desired = self.load_image()
        self.assertTrue(utils.is_batched_image(actual))
        self.assertImagesAlmostEqual(actual, desired)

    def test_read_image_resize(self):
        image_size = (200, 300)
        actual = io.read_image(self.default_image_file(), size=image_size)
        desired = self.load_image(backend="PIL").resize(image_size[::-1])
        self.assertImagesAlmostEqual(actual, desired)

    def test_read_image_resize_scalar(self):
        edge_size = 200

        image = self.load_image(backend="PIL")
        aspect_ratio = utils.calculate_aspect_ratio((image.height, image.width))
        image_size = utils.edge_to_image_size(edge_size, aspect_ratio)

        actual = io.read_image(self.default_image_file(), size=edge_size)
        desired = image.resize(image_size[::-1])
        self.assertImagesAlmostEqual(actual, desired)

    def test_read_image_resize_other(self):
        with self.assertRaises(TypeError):
            io.read_image(self.default_image_file(), size="invalid_size")

    def test_write_image(self):
        torch.manual_seed(0)
        image = torch.rand(3, 100, 100)
        with get_tmp_dir() as tmp_dir:
            file = path.join(tmp_dir, "tmp_image.png")
            io.write_image(image, file)

            actual = self.load_image(file=file)

        desired = image
        self.assertImagesAlmostEqual(actual, desired)

    @mock.patch("pystiche.image.io._show_pil_image")
    def test_show_image_smoke(self, show_pil_image_mock):
        image = self.load_image()
        io.show_image(image)
        io.show_image(self.default_image_file())
        with self.assertRaises(TypeError):
            io.show_image(None)
        io.show_image(image, size=100)
        io.show_image(image, size=(100, 200))


class TestUtils(PysticheTestCase):
    def test_verify_is_single_image(self):
        single_image = torch.zeros(1, 1, 1)
        utils.verify_is_single_image(single_image)

        for dtype in (torch.uint8, torch.int):
            with self.assertRaises(TypeError):
                image = single_image.clone().to(dtype)
                utils.verify_is_single_image(image)

        for dim in (2, 4):
            with self.assertRaises(TypeError):
                image = torch.tensor(*[0.0] * dim)
                utils.verify_is_single_image(image)

    def test_is_single_image(self):
        single_image = torch.zeros(1, 1, 1)
        self.assertTrue(utils.is_single_image(single_image))
        self.assertFalse(utils.is_single_image(single_image.byte()))
        self.assertFalse(utils.is_single_image(single_image.unsqueeze(0)))

    def test_verify_is_batched_image(self):
        batched_image = torch.zeros(1, 1, 1, 1)
        utils.verify_is_batched_image(batched_image)

        for dtype in (torch.uint8, torch.int):
            with self.assertRaises(TypeError):
                image = batched_image.clone().to(dtype)
                utils.verify_is_batched_image(image)

        for dim in (3, 5):
            with self.assertRaises(TypeError):
                image = torch.tensor(*[0.0] * dim)
                utils.verify_is_batched_image(image)

    def test_is_batched_image(self):
        batched_image = torch.zeros(1, 1, 1, 1)
        self.assertTrue(utils.is_batched_image(batched_image))
        self.assertFalse(utils.is_batched_image(batched_image.byte()))
        self.assertFalse(utils.is_batched_image(batched_image.squeeze(0)))

    def test_verify_is_image(self):
        single_image = torch.zeros(1, 1, 1)
        utils.verify_is_image(single_image)
        batched_image = single_image.unsqueeze(0)
        utils.verify_is_image(batched_image)

        with self.assertRaises(TypeError):
            utils.verify_is_image(None)

        for dtype in (torch.uint8, torch.int):
            with self.assertRaises(TypeError):
                image = torch.empty([1] * 3, dtype=dtype)
                utils.verify_is_image(image)

        for dim in (2, 5):
            with self.assertRaises(TypeError):
                image = torch.empty([1] * dim)
                utils.verify_is_image(image)

    def test_is_image(self):
        single_image = torch.zeros(1, 1, 1)
        batched_image = single_image.unsqueeze(0)
        self.assertTrue(utils.is_image(single_image))
        self.assertTrue(utils.is_image(batched_image))
        self.assertFalse(utils.is_image(single_image.byte()))
        self.assertFalse(utils.is_image(batched_image.byte()))

    def test_is_image_size(self):
        image_size = [1, 1]
        self.assertTrue(utils.is_image_size(image_size))
        self.assertTrue(utils.is_image_size(tuple(image_size)))
        self.assertFalse(utils.is_image_size(image_size[0]))
        self.assertFalse(utils.is_image_size(image_size + image_size))
        self.assertFalse(
            utils.is_image_size([float(edge_size) for edge_size in image_size])
        )

    def test_is_edge_size(self):
        edge_size = 1
        self.assertTrue(utils.is_edge_size(edge_size))
        self.assertFalse(utils.is_edge_size(float(edge_size)))
        self.assertFalse(utils.is_edge_size((edge_size, edge_size)))

    def test_calculate_aspect_ratio(self):
        height = 2
        width = 3
        image_size = (height, width)

        actual = utils.calculate_aspect_ratio(image_size)
        desired = width / height
        self.assertAlmostEqual(actual, desired)

    def test_image_to_edge_size(self):
        image_size = (1, 2)

        edges = ("short", "long", "vert", "horz")
        actual = tuple(utils.image_to_edge_size(image_size, edge) for edge in edges)
        desired = (1, 2, 1, 2)
        self.assertTupleEqual(actual, desired)

    def test_edge_to_image_size_short(self):
        edge_size = 2
        edge = "short"

        aspect_ratio = 2.0
        actual = utils.edge_to_image_size(edge_size, aspect_ratio, edge)
        desired = (edge_size, round(edge_size * aspect_ratio))
        self.assertTupleEqual(actual, desired)

        aspect_ratio = 0.5
        actual = utils.edge_to_image_size(edge_size, aspect_ratio, edge)
        desired = (round(edge_size / aspect_ratio), edge_size)
        self.assertTupleEqual(actual, desired)

    def test_calculate_resized_image_size_long(self):
        edge_size = 2
        edge = "long"

        aspect_ratio = 2.0
        actual = utils.edge_to_image_size(edge_size, aspect_ratio, edge)
        desired = (round(edge_size / aspect_ratio), edge_size)
        self.assertTupleEqual(actual, desired)

        aspect_ratio = 0.5
        actual = utils.edge_to_image_size(edge_size, aspect_ratio, edge)
        desired = (edge_size, round(edge_size * aspect_ratio))
        self.assertTupleEqual(actual, desired)

    def test_edge_to_image_size_vert_horz(self):
        aspect_ratio = 2.0
        edge_size = 2

        actual = utils.edge_to_image_size(edge_size, aspect_ratio, edge="vert")
        desired = (edge_size, round(edge_size * aspect_ratio))
        self.assertTupleEqual(actual, desired)

        actual = utils.edge_to_image_size(edge_size, aspect_ratio, edge="horz")
        desired = (round(edge_size * aspect_ratio), edge_size)
        self.assertTupleEqual(actual, desired)

    def test_extract_batch_size(self):
        batch_size = 3

        batched_image = torch.zeros(batch_size, 1, 1, 1)
        actual = utils.extract_batch_size(batched_image)
        desired = batch_size
        self.assertEqual(actual, desired)

        single_image = torch.zeros(1, 1, 1)
        with self.assertRaises(TypeError):
            utils.extract_batch_size(single_image)

    def test_extract_num_channels(self):
        num_channels = 3

        single_image = torch.zeros(num_channels, 1, 1)
        actual = utils.extract_num_channels(single_image)
        desired = num_channels
        self.assertEqual(actual, desired)

        batched_image = single_image.unsqueeze(0)
        actual = utils.extract_num_channels(batched_image)
        desired = num_channels
        self.assertEqual(actual, desired)

    def test_extract_image_size(self):
        height = 2
        width = 3
        image = torch.empty(1, 1, height, width)

        actual = utils.extract_image_size(image)
        desired = (height, width)
        self.assertTupleEqual(actual, desired)

    def test_extract_edge_size(self):
        height = 2
        width = 3
        image = torch.empty(1, 1, height, width)

        edges = ("short", "long", "vert", "horz")
        actual = tuple(utils.extract_edge_size(image, edge=edge) for edge in edges)
        desired = (height, width, height, width)
        self.assertTupleEqual(actual, desired)

    def test_extract_aspect_ratio(self):
        height = 2
        width = 3
        image = torch.empty(1, 1, height, width)

        actual = utils.extract_aspect_ratio(image)
        desired = width / height
        self.assertAlmostEqual(actual, desired)

    def test_make_batched_image(self):
        single_image = torch.empty(1, 1, 1)
        batched_image = utils.make_batched_image(single_image)
        self.assertTrue(utils.is_batched_image(batched_image))

    def test_make_single_image(self):
        batched_image = torch.empty(1, 1, 1, 1)
        single_image = utils.make_single_image(batched_image)
        self.assertTrue(utils.is_single_image(single_image))

        batched_image = torch.empty(2, 1, 1, 1)
        with self.assertRaises(RuntimeError):
            utils.make_single_image(batched_image)

    def test_force_image(self):
        @utils.force_image
        def identity(image):
            return image

        single_image = torch.empty(1, 1, 1)
        batched_image = torch.empty(1, 1, 1, 1)

        self.assertIs(identity(single_image), single_image)
        self.assertIs(identity(batched_image), batched_image)

        with self.assertRaises(TypeError):
            identity(None)

    def test_force_single_image(self):
        @utils.force_single_image
        def identity(single_image):
            self.assertTrue(utils.is_single_image(single_image))
            return single_image

        single_image = torch.empty(1, 1, 1)
        batched_image = torch.empty(1, 1, 1, 1)

        self.assertIs(identity(single_image), single_image)
        self.assertTensorAlmostEqual(identity(batched_image), batched_image)

        with self.assertRaises(TypeError):
            identity(None)

    def test_force_batched_image(self):
        @utils.force_batched_image
        def identity(batched_image):
            self.assertTrue(utils.is_batched_image(batched_image))
            return batched_image

        single_image = torch.empty(1, 1, 1)
        batched_image = torch.empty(1, 1, 1, 1)

        self.assertTensorAlmostEqual(identity(single_image), single_image)
        self.assertIs(identity(batched_image), batched_image)

        with self.assertRaises(TypeError):
            identity(None)
