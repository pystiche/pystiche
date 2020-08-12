import itertools
import os
import shutil
from os import path

import pytest
import pytorch_testing_utils as ptu

import torch
from torch import nn

from pystiche import data
from pystiche.image import read_image, write_image


def test_DownloadableImage_generate_file(subtests, test_image_url):
    titles = (None, "girl with painted face")
    authors = (None, "Steve Kelly")
    titles_and_authors = itertools.product(titles, authors)

    desireds = (
        "test_image.png",
        "steve_kelly.png",
        "girl_with_painted_face.png",
        "girl_with_painted_face__steve_kelly.png",
    )

    for (title, author), desired in zip(titles_and_authors, desireds):
        with subtests.test(title=title, author=author):
            actual = data.DownloadableImage.generate_file(
                test_image_url, title=title, author=author
            )
            assert actual == desired


def test_DownloadableImage_download(tmpdir, test_image_url, test_image):
    image = data.DownloadableImage(test_image_url)
    image.download(tmpdir)

    file = path.join(tmpdir, image.file)
    assert path.exists(file)

    actual = read_image(file)
    desired = test_image
    ptu.assert_allclose(actual, desired)


def test_DownloadableImage_download_guides(tmpdir, test_image_url, test_image):
    guide = data.DownloadableImage(
        test_image_url, file="guide" + path.splitext(test_image_url)[1],
    )
    image = data.DownloadableImage(
        test_image_url, guides=data.DownloadableImageCollection({"guide": guide}),
    )
    image.download(tmpdir)

    actual = read_image(path.join(tmpdir, guide.file))
    desired = test_image
    ptu.assert_allclose(actual, desired)


def test_DownloadableImage_download_exist(
    tmpdir, test_image_url, test_image_file, test_image
):
    image = data.DownloadableImage(test_image_url)
    file = path.join(tmpdir, image.file)

    shutil.copyfile(test_image_file, file)

    with pytest.raises(FileExistsError):
        image.download(tmpdir)

    image.md5 = "invalid_hash"
    with pytest.raises(FileExistsError):
        image.download(tmpdir)

    image.md5 = "a858d33c424eaac1322cf3cab6d3d568"
    image.download(tmpdir)


def test_DownloadableImage_download_overwrite(tmpdir, test_image_url, test_image):
    def create_fake_image(file):
        open(file, "wb").close()

    image = data.DownloadableImage(test_image_url)
    file = path.join(tmpdir, image.file)

    create_fake_image(file)
    image.download(tmpdir, overwrite=True)

    actual = read_image(file)
    desired = test_image
    ptu.assert_allclose(actual, desired)

    create_fake_image(file)
    image.md5 = "a858d33c424eaac1322cf3cab6d3d568"
    image.download(tmpdir, overwrite=True)

    actual = read_image(file)
    desired = test_image
    ptu.assert_allclose(actual, desired)


def test_DownloadableImage_read(tmpdir, test_image_url, test_image):
    image = data.DownloadableImage(test_image_url)

    actual = image.read(tmpdir)
    desired = test_image
    ptu.assert_allclose(actual, desired)


def test_DownloadableImage_repr_smoke(test_image_url):
    image = data.DownloadableImage(test_image_url)
    assert isinstance(repr(image), str)


def test_DownloadableImageCollection(tmpdir, test_image_url, test_image):
    images = {"test_image": data.DownloadableImage(test_image_url)}
    collection = data.DownloadableImageCollection(images,)
    collection.download(root=tmpdir)

    actual = collection["test_image"].read(root=tmpdir)
    desired = test_image
    ptu.assert_allclose(actual, desired)


def test_DownloadableImageCollection_read(tmpdir, test_image_url, test_image):
    names = [str(idx) for idx in range(3)]
    collection = data.DownloadableImageCollection(
        {name: data.DownloadableImage(test_image_url) for name in names}
    )
    images = collection.read(tmpdir)

    ptu.assert_allclose(images, dict(zip(names, [test_image] * len(names))))


def test_LocalImage_collect_guides(tmpdir, test_image_file):
    def create_guides(root, file):
        dir = path.join(root, path.splitext(file)[0])
        os.makedirs(dir)
        src = test_image_file
        files = {}
        for idx in range(3):
            region = str(idx)
            dst = path.join(dir, region + path.splitext(src)[1])
            shutil.copyfile(src, dst)
            files[region] = dst

        return files

    file = "image.jpg"
    files = create_guides(tmpdir, file)

    image = data.LocalImage(path.join(tmpdir, file), collect_local_guides=True)
    assert image.guides is not None

    actual = {region: guide.file for region, guide in image.guides}
    desired = files
    assert actual == desired


def test_LocalImage_collect_guides_no_dir(tmpdir):
    image = data.LocalImage(path.join(tmpdir, "image.jpg"), collect_local_guides=True)
    assert image.guides is None


def test_LocalImage_collect_guides_empty_dir(tmpdir):
    file = "image.jpg"
    dir = path.join(tmpdir, path.splitext(file)[0])
    os.mkdir(dir)
    open(path.join(dir, "not_an_image.txt"), "wb").close()

    image = data.LocalImage(path.join(tmpdir, file), collect_local_guides=True)

    assert image.guides is None


def test_LocalImage_read_abs_path(test_image_file, test_image):
    image = data.LocalImage(test_image_file)

    actual = image.read()
    desired = test_image
    ptu.assert_allclose(actual, desired)

    actual = image.read(root="/invalid/root")
    desired = test_image
    ptu.assert_allclose(actual, desired)


def test_LocalImage_read_rel_path(test_image_file, test_image):
    root, filename = path.dirname(test_image_file), path.basename(test_image_file)
    image = data.LocalImage(filename)

    actual = image.read(root)
    desired = test_image
    ptu.assert_allclose(actual, desired)


def test_LocalImage_repr_smoke():
    image = data.LocalImage("image", transform=nn.Module(), note="note")
    assert isinstance(repr(image), str)


def test_LocalImageCollection_len():
    num_images = 3
    images = {str(idx): data.LocalImage(str(idx)) for idx in range(num_images)}
    collection = data.LocalImageCollection(images)

    actual = len(collection)
    desired = num_images
    assert actual == desired


def test_LocalImageCollection_getitem():
    images = {str(idx): data.LocalImage(str(idx)) for idx in range(3)}
    collection = data.LocalImageCollection(images)

    actual = {str(idx): collection[str(idx)] for idx in range(len(collection))}
    desired = images
    assert actual == desired


def test_LocalImageCollection_iter():
    images = {str(idx): data.LocalImage(str(idx)) for idx in range(3)}
    collection = data.LocalImageCollection(images)

    actual = {name: image for name, image in collection}
    desired = images
    assert actual == desired


def test_LocalImageCollection_read(tmpdir):
    def create_images(root):
        torch.manual_seed(0)
        files = {}
        for idx in range(3):
            name = str(idx)
            image = torch.rand(1, 3, 32, 32)
            file = path.join(root, f"{name}.png")
            write_image(image, file)
            files[name] = file
        return files

    files = create_images(tmpdir)
    collection = data.LocalImageCollection(
        {name: data.LocalImage(file) for name, file in files.items()}
    )

    actual = collection.read()
    desired = {name: read_image(file) for name, file in files.items()}
    ptu.assert_allclose(actual, desired)


def test_LocalImageCollection_repr_smoke():
    images = {str(idx): data.LocalImage(str(idx)) for idx in range(3)}
    collection = data.LocalImageCollection(images)

    assert isinstance(repr(collection), str)
