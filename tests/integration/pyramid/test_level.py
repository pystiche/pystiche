from pystiche import pyramid


def test_PyramidLevel_iter():
    edge_size = 1
    num_steps = 100
    edge = "short"

    pyramid_level = pyramid.PyramidLevel(edge_size, num_steps, edge)

    actual = tuple(iter(pyramid_level))
    desired = tuple(range(1, num_steps + 1))
    assert actual == desired
