import unittest
import torch
from pystiche.image import utils


class Tester(unittest.TestCase):
    def test_verify_is_image(self):
        single_image = torch.zeros(1, 1, 1)
        utils.verify_is_image(single_image)
        batched_image = single_image.unsqueeze(0)
        utils.verify_is_image(batched_image)

        for dtype in (torch.uint8, torch.int):
            with self.assertRaises(TypeError):
                image = torch.zeros(1, 1, 1, dtype=dtype)
                utils.verify_is_image(image)

        for dim in (2, 5):
            with self.assertRaises(TypeError):
                image = torch.tensor(*[0.0] * dim)
                utils.verify_is_image(image)

    def test_is_single_image(self):
        single_image = torch.zeros(1, 1, 1)
        self.assertTrue(utils.is_single_image(single_image))
        self.assertFalse(utils.is_single_image(single_image.byte()))
        self.assertFalse(utils.is_single_image(single_image.unsqueeze(0)))

    def test_is_batched_image(self):
        batched_image = torch.zeros(1, 1, 1, 1)
        self.assertTrue(utils.is_batched_image(batched_image))
        self.assertFalse(utils.is_batched_image(batched_image.byte()))
        self.assertFalse(utils.is_batched_image(batched_image.squeeze(0)))

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
        actual = tuple([utils.image_to_edge_size(image_size, edge) for edge in edges])
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
        with self.assertRaises(RuntimeError):
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
        actual = tuple([utils.extract_edge_size(image, edge=edge) for edge in edges])
        desired = (height, width, height, width)
        self.assertTupleEqual(actual, desired)

    def test_extract_aspect_ratio(self):
        height = 2
        width = 3
        image = torch.empty(1, 1, height, width)

        actual = utils.extract_aspect_ratio(image)
        desired = width / height
        self.assertAlmostEqual(actual, desired)


if __name__ == "__main__":
    unittest.main()
