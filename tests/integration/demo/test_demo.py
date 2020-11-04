import pytest

from pystiche import data, demo


def test_demo_images_smoke():
    images = demo.demo_images()
    assert isinstance(images, data.DownloadableImageCollection)

    names, _ = zip(*iter(images))
    assert set(names) == {
        "bird1",
        "paint",
        "bird2",
        "mosaic",
        "castle",
        "church",
        "cliff",
    }


@pytest.mark.parametrize(
    ("name", "regions"),
    [
        ("castle", ("building", "water", "sky")),
        ("church", ("building", "sky")),
        ("cliff", ("water",)),
    ],
)
def test_demo_images_guides_smoke(name, regions):
    image = demo.demo_images()[name]

    assert isinstance(image.guides, data.DownloadableImageCollection)

    names, _ = zip(*iter(image.guides))
    assert set(names).issuperset(set(regions))
