import os
from os import path

import pytest

from pystiche import data


def make_fake_image(root, name, ext=".jpg"):
    if not path.exists(root):
        os.mkdir(root)
    file = path.join(root, name + ext)
    with open(file, "wb"):
        pass
    return file


@pytest.fixture
def image_folder(tmpdir):
    tmpdir = str(tmpdir)
    make_fake_image(tmpdir, "image0")

    dir1 = path.join(tmpdir, "dir1")
    make_fake_image(dir1, "image1")

    dir2 = path.join(dir1, "dir2")
    make_fake_image(dir2, "image2")
    return tmpdir


def test_walkupto(image_folder):
    actual = tuple(data.datasets.walkupto(image_folder))
    desired = (
        (image_folder, ["dir1"], ["image0.jpg"]),
        (path.join(image_folder, "dir1"), ["dir2"], ["image1.jpg"]),
        (path.join(image_folder, "dir1", "dir2"), [], ["image2.jpg"]),
    )
    assert actual == desired


def test_walkupto_depth(image_folder):
    actual = tuple(data.datasets.walkupto(image_folder, depth=1))
    desired = (
        (image_folder, ["dir1"], ["image0.jpg"]),
        (path.join(image_folder, "dir1"), ["dir2"], ["image1.jpg"]),
    )
    assert actual == desired


class TestImageFolderDataset:
    def test_core(self, image_folder):
        dataset = data.ImageFolderDataset(image_folder)

        actual = len(dataset)
        desired = 3
        assert actual == desired

        actual = dataset.image_files
        desired = (
            path.join(image_folder, "image0.jpg"),
            path.join(image_folder, "dir1", "image1.jpg"),
            path.join(image_folder, "dir1", "dir2", "image2.jpg"),
        )
        assert actual == desired

    def test_depth(self, image_folder):
        dataset = data.ImageFolderDataset(image_folder, depth=1)

        actual = len(dataset)
        desired = 2
        assert actual == desired

        actual = dataset.image_files
        desired = (
            path.join(image_folder, "image0.jpg"),
            path.join(image_folder, "dir1", "image1.jpg"),
        )
        assert actual == desired

    def test_getitem(self, image_folder):
        def transform(file):
            return path.join(path.dirname(file), f"transformed_{path.basename(file)}")

        def importer(file):
            return path.join(path.dirname(file), f"imported_{path.basename(file)}")

        dataset = data.ImageFolderDataset(
            image_folder, transform=transform, importer=importer
        )

        actual = dataset[0]
        desired = path.join(image_folder, "transformed_imported_image0.jpg")
        assert actual == desired
