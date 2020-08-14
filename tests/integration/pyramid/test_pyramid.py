import torch

from pystiche import ops, pyramid
from pystiche.image import (
    calculate_aspect_ratio,
    edge_to_image_size,
    extract_image_size,
)


def test_ImagePyramid_build_levels_scalar_num_steps():
    num_levels = 3
    num_steps = 2

    pyramid_levels = pyramid.ImagePyramid([1] * num_levels, num_steps)

    actual = tuple(level.num_steps for level in pyramid_levels)
    desired = tuple([num_steps] * num_levels)
    assert actual == desired


def test_ImagePyramid_build_levels_scalar_edge():
    num_levels = 3
    edge = "long"

    pyramid_levels = pyramid.ImagePyramid([1] * num_levels, 1, edge=edge)

    actual = tuple(level.edge for level in pyramid_levels)
    desired = tuple([edge] * num_levels)
    assert actual == desired


def test_ImagePyramid_len():
    num_levels = 3

    image_pyramid = pyramid.ImagePyramid([1] * num_levels, [1] * num_levels)

    actual = len(image_pyramid)
    desired = num_levels
    assert actual == desired


def test_ImagePyramid_getitem():
    edge_sizes = (1, 2, 3)
    num_steps = (4, 5, 6)

    image_pyramid = pyramid.ImagePyramid(edge_sizes, num_steps)

    for idx, desired in enumerate(zip(edge_sizes, num_steps)):
        pyramid_level = image_pyramid[idx]
        assert isinstance(pyramid_level, pyramid.PyramidLevel)

        actual = (pyramid_level.edge_size, pyramid_level.num_steps)
        assert actual == desired


def test_ImagePyramid_iter():
    edge_sizes = (1, 2, 3)
    num_steps = (4, 5, 6)

    image_pyramid = pyramid.ImagePyramid(edge_sizes, num_steps)

    desireds = zip(edge_sizes, num_steps)
    for pyramid_level, desired in zip(image_pyramid, desireds):
        assert isinstance(pyramid_level, pyramid.PyramidLevel)

        actual = (pyramid_level.edge_size, pyramid_level.num_steps)
        assert actual == desired


def test_ImagePyramid_iter_resize(subtests):
    class TestOperator(ops.PixelComparisonOperator):
        def target_image_to_repr(self, image):
            return image, None

        def input_image_to_repr(self, image, ctx):
            pass

        def calculate_score(self, input_repr, target_repr, ctx):
            pass

    initial_image_size = (5, 4)
    edge_sizes = (2, 4)

    torch.manual_seed(0)
    target_guide = torch.rand((1, 3, *initial_image_size))
    target_image = torch.rand((1, 3, *initial_image_size))
    input_guide = torch.rand((1, 3, *initial_image_size))

    aspect_ratio = calculate_aspect_ratio(initial_image_size)
    image_sizes = [
        edge_to_image_size(edge_size, aspect_ratio) for edge_size in edge_sizes
    ]

    op = TestOperator()
    op.set_target_guide(target_guide)
    op.set_target_image(target_image)
    op.set_input_guide(input_guide)

    image_pyramid = pyramid.ImagePyramid(edge_sizes, 1, resize_targets=(op,))
    for pyramid_level, image_size in zip(image_pyramid, image_sizes):
        for attr in ("target_guide", "target_image", "input_guide"):
            with subtests.test(attr, pyramid_level=pyramid_level):
                actual = extract_image_size(getattr(op, attr))
                desired = image_size
                assert actual == desired


def test_ImagePyramid_iter_restore():
    class TestOperator(ops.PixelComparisonOperator):
        def target_image_to_repr(self, image):
            return image, None

        def input_image_to_repr(self, image, ctx):
            pass

        def calculate_score(self, input_repr, target_repr, ctx):
            pass

    torch.manual_seed(0)
    size = (1, 3, 128, 128)
    image = torch.rand(*size)

    op = TestOperator()
    op.set_target_image(image)

    image_pyramid = pyramid.ImagePyramid((1,), 1, resize_targets=(op,))
    try:
        for _ in image_pyramid:
            raise Exception
    except Exception:
        pass

    actual = op.target_image.size()
    desired = size
    assert actual == desired


def test_OctaveImagePyramid():
    max_edge_size = 16
    min_edge_size = 2

    image_pyramid = pyramid.OctaveImagePyramid(
        max_edge_size, 1, min_edge_size=min_edge_size
    )
    assert len(image_pyramid) == 4

    for idx in range(len(image_pyramid)):
        actual = image_pyramid[idx].edge_size
        desired = 2 ** (idx + 1)
        assert actual == desired


def test_OctaveImagePyramid_num_levels():
    max_edge_size = 16
    num_levels = 3
    min_edge_size = 2

    image_pyramid = pyramid.OctaveImagePyramid(
        max_edge_size, 1, num_levels=num_levels, min_edge_size=min_edge_size
    )
    assert len(image_pyramid) == 3

    for idx in range(len(image_pyramid)):
        actual = image_pyramid[idx].edge_size
        desired = 2 ** (idx + 2)
        assert actual == desired
