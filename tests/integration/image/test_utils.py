import pytest
import pytorch_testing_utils as ptu

import torch

from pystiche import image as image_


def test_verify_is_single_image():
    single_image = torch.zeros(1, 1, 1)
    image_.verify_is_single_image(single_image)

    for dtype in (torch.uint8, torch.int):
        with pytest.raises(TypeError):
            image = single_image.clone().to(dtype)
            image_.verify_is_single_image(image)

    for dim in (2, 4):
        with pytest.raises(TypeError):
            image = torch.tensor(*[0.0] * dim)
            image_.verify_is_single_image(image)


def test_is_single_image():
    single_image = torch.zeros(1, 1, 1)
    assert image_.is_single_image(single_image)
    assert not image_.is_single_image(single_image.byte())
    assert not image_.is_single_image(single_image.unsqueeze(0))


def test_verify_is_batched_image():
    batched_image = torch.zeros(1, 1, 1, 1)
    image_.verify_is_batched_image(batched_image)

    for dtype in (torch.uint8, torch.int):
        with pytest.raises(TypeError):
            image = batched_image.clone().to(dtype)
            image_.verify_is_batched_image(image)

    for dim in (3, 5):
        with pytest.raises(TypeError):
            image = torch.tensor(*[0.0] * dim)
            image_.verify_is_batched_image(image)


def test_is_batched_image():
    batched_image = torch.zeros(1, 1, 1, 1)
    assert image_.is_batched_image(batched_image)
    assert not image_.is_batched_image(batched_image.byte())
    assert not image_.is_batched_image(batched_image.squeeze(0))


def test_verify_is_image():
    single_image = torch.zeros(1, 1, 1)
    image_.verify_is_image(single_image)
    batched_image = single_image.unsqueeze(0)
    image_.verify_is_image(batched_image)

    with pytest.raises(TypeError):
        image_.verify_is_image(None)

    for dtype in (torch.uint8, torch.int):
        with pytest.raises(TypeError):
            image = torch.empty([1] * 3, dtype=dtype)
            image_.verify_is_image(image)

    for dim in (2, 5):
        with pytest.raises(TypeError):
            image = torch.empty([1] * dim)
            image_.verify_is_image(image)


def test_is_image():
    single_image = torch.zeros(1, 1, 1)
    batched_image = single_image.unsqueeze(0)
    assert image_.is_image(single_image)
    assert image_.is_image(batched_image)
    assert not image_.is_image(single_image.byte())
    assert not image_.is_image(batched_image.byte())


def test_is_image_size():
    image_size = [1, 1]
    assert image_.is_image_size(image_size)
    assert image_.is_image_size(tuple(image_size))
    assert not image_.is_image_size(image_size[0])
    assert not image_.is_image_size(image_size + image_size)
    assert not image_.is_image_size([float(edge_size) for edge_size in image_size])


def test_is_edge_size():
    edge_size = 1
    assert image_.is_edge_size(edge_size)
    assert not image_.is_edge_size(float(edge_size))
    assert not image_.is_edge_size((edge_size, edge_size))


def test_calculate_aspect_ratio():
    height = 2
    width = 3
    image_size = (height, width)

    actual = image_.calculate_aspect_ratio(image_size)
    desired = width / height
    assert actual == pytest.approx(desired)


def test_image_to_edge_size():
    image_size = (1, 2)

    edges = ("short", "long", "vert", "horz")
    actual = tuple(image_.image_to_edge_size(image_size, edge) for edge in edges)
    desired = (1, 2, 1, 2)
    assert actual == desired


def test_edge_to_image_size_short():
    edge_size = 2
    edge = "short"

    aspect_ratio = 2.0
    actual = image_.edge_to_image_size(edge_size, aspect_ratio, edge)
    desired = (edge_size, round(edge_size * aspect_ratio))
    assert actual == desired

    aspect_ratio = 0.5
    actual = image_.edge_to_image_size(edge_size, aspect_ratio, edge)
    desired = (round(edge_size / aspect_ratio), edge_size)
    assert actual == desired


def test_calculate_resized_image_size_long():
    edge_size = 2
    edge = "long"

    aspect_ratio = 2.0
    actual = image_.edge_to_image_size(edge_size, aspect_ratio, edge)
    desired = (round(edge_size / aspect_ratio), edge_size)
    assert actual == desired

    aspect_ratio = 0.5
    actual = image_.edge_to_image_size(edge_size, aspect_ratio, edge)
    desired = (edge_size, round(edge_size * aspect_ratio))
    assert actual == desired


def test_edge_to_image_size_vert_horz():
    aspect_ratio = 2.0
    edge_size = 2

    actual = image_.edge_to_image_size(edge_size, aspect_ratio, edge="vert")
    desired = (edge_size, round(edge_size * aspect_ratio))
    assert actual == desired

    actual = image_.edge_to_image_size(edge_size, aspect_ratio, edge="horz")
    desired = (round(edge_size / aspect_ratio), edge_size)
    assert actual == desired


def test_extract_batch_size():
    batch_size = 3

    batched_image = torch.zeros(batch_size, 1, 1, 1)
    actual = image_.extract_batch_size(batched_image)
    desired = batch_size
    assert actual == desired

    single_image = torch.zeros(1, 1, 1)
    with pytest.raises(TypeError):
        image_.extract_batch_size(single_image)


def test_extract_num_channels():
    num_channels = 3

    single_image = torch.zeros(num_channels, 1, 1)
    actual = image_.extract_num_channels(single_image)
    desired = num_channels
    assert actual == desired

    batched_image = single_image.unsqueeze(0)
    actual = image_.extract_num_channels(batched_image)
    desired = num_channels
    assert actual == desired


def test_extract_image_size():
    height = 2
    width = 3
    image = torch.empty(1, 1, height, width)

    actual = image_.extract_image_size(image)
    desired = (height, width)
    assert actual == desired


def test_extract_edge_size():
    height = 2
    width = 3
    image = torch.empty(1, 1, height, width)

    edges = ("short", "long", "vert", "horz")
    actual = tuple(image_.extract_edge_size(image, edge=edge) for edge in edges)
    desired = (height, width, height, width)
    assert actual == desired


def test_extract_aspect_ratio():
    height = 2
    width = 3
    image = torch.empty(1, 1, height, width)

    actual = image_.extract_aspect_ratio(image)
    desired = width / height
    assert actual == pytest.approx(desired)


class TestMakeImage:
    def test_batched_image(self):
        single_image = torch.empty(1, 1, 1)
        batched_image = image_.make_batched_image(single_image)
        assert image_.is_batched_image(batched_image)

    def test_single_image(self):
        batched_image = torch.empty(1, 1, 1, 1)
        single_image = image_.make_single_image(batched_image)
        assert image_.is_single_image(single_image)

        batched_image = torch.empty(2, 1, 1, 1)
        with pytest.raises(RuntimeError):
            image_.make_single_image(batched_image)


class TestForceImage:
    def test_image(self):
        @image_.force_image
        def identity(image):
            return image

        single_image = torch.empty(1, 1, 1)
        batched_image = torch.empty(1, 1, 1, 1)

        assert identity(single_image) is single_image
        assert identity(batched_image) is batched_image

        with pytest.raises(TypeError):
            identity(None)

    def test_single_image(self):
        @image_.force_single_image
        def identity(single_image):
            assert image_.is_single_image(single_image)
            return single_image

        single_image = torch.empty(1, 1, 1)
        batched_image = torch.empty(1, 1, 1, 1)

        assert identity(single_image) is single_image
        ptu.assert_allclose(identity(batched_image), batched_image)

        with pytest.raises(TypeError):
            identity(None)

    def test_batched_image(self):
        @image_.force_batched_image
        def identity(batched_image):
            assert image_.is_batched_image(batched_image)
            return batched_image

        single_image = torch.empty(1, 1, 1)
        batched_image = torch.empty(1, 1, 1, 1)

        ptu.assert_allclose(identity(single_image), single_image)
        assert identity(batched_image) is batched_image

        with pytest.raises(TypeError):
            identity(None)
