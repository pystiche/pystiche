import contextlib
import itertools
import os
import shutil
from os import path
from shutil import copyfile

import torch
from torch import nn

from pystiche import data
from pystiche.image import read_image, write_image

from .utils import PysticheTestCase, get_tmp_dir


class TestDatasets(PysticheTestCase):
    @staticmethod
    def create_fake_image(name, ext=".jpg"):
        file = name + ext
        with open(file, "wb"):
            pass
        return file

    @staticmethod
    @contextlib.contextmanager
    def create_fake_image_folder():
        with get_tmp_dir() as top:
            TestDatasets.create_fake_image(path.join(top, "image0"))

            dir1 = path.join(top, "dir1")
            os.mkdir(dir1)
            TestDatasets.create_fake_image(path.join(dir1, "image1"))

            dir2 = path.join(dir1, "dir2")
            os.mkdir(dir2)
            TestDatasets.create_fake_image(path.join(dir2, "image2"))

            yield top

    def test_walkupto(self):
        with self.create_fake_image_folder() as top:
            actual = tuple(data.datasets.walkupto(top))
            desired = (
                (top, ["dir1"], ["image0.jpg"]),
                (path.join(top, "dir1"), ["dir2"], ["image1.jpg"]),
                (path.join(top, "dir1", "dir2"), [], ["image2.jpg"]),
            )
            self.assertEqual(actual, desired)

    def test_walkupto_depth(self):
        with self.create_fake_image_folder() as top:
            actual = tuple(data.datasets.walkupto(top, depth=1))
            desired = (
                (top, ["dir1"], ["image0.jpg"]),
                (path.join(top, "dir1"), ["dir2"], ["image1.jpg"]),
            )
            self.assertEqual(actual, desired)

    def test_ImageFolderDataset(self):
        with self.create_fake_image_folder() as root:
            dataset = data.ImageFolderDataset(root)

            actual = len(dataset)
            desired = 3
            self.assertEqual(actual, desired)

            actual = dataset.image_files
            desired = (
                path.join(root, "image0.jpg"),
                path.join(root, "dir1", "image1.jpg"),
                path.join(root, "dir1", "dir2", "image2.jpg"),
            )
            self.assertEqual(actual, desired)

    def test_ImageFolderDataset_depth(self):
        with self.create_fake_image_folder() as root:
            dataset = data.ImageFolderDataset(root, depth=1)

            actual = len(dataset)
            desired = 2
            self.assertEqual(actual, desired)

            actual = dataset.image_files
            desired = (
                path.join(root, "image0.jpg"),
                path.join(root, "dir1", "image1.jpg"),
            )
            self.assertEqual(actual, desired)

    def test_ImageFolderDataset_getitem(self):
        def transform(file):
            return path.join(path.dirname(file), f"transformed_{path.basename(file)}")

        def importer(file):
            return path.join(path.dirname(file), f"imported_{path.basename(file)}")

        with self.create_fake_image_folder() as root:
            dataset = data.ImageFolderDataset(
                root, transform=transform, importer=importer
            )

            actual = dataset[0]
            desired = path.join(root, "transformed_imported_image0.jpg")
            self.assertEqual(actual, desired)


class TestLicense(PysticheTestCase):
    def test_UnknownLicense_repr_smoke(self):
        license = data.UnknownLicense()
        self.assertIsInstance(repr(license), str)

    def test_NoLicense_repr_smoke(self):
        license = data.NoLicense()
        self.assertIsInstance(repr(license), str)

    def test_PublicDomainLicense_repr_smoke(self):
        license = data.PublicDomainLicense()
        self.assertIsInstance(repr(license), str)

    def test_ExpiredCopyrightLicense_repr_smoke(self):
        license = data.ExpiredCopyrightLicense(1970)
        self.assertIsInstance(repr(license), str)

    def test_Pixabay_repr_smoke(self):
        license = data.PixabayLicense()
        self.assertIsInstance(repr(license), str)

    def test_CreativeCommonsLicense_repr_smoke(self):
        license = data.CreativeCommonsLicense(("by", "sa"), "3.0")
        self.assertIsInstance(repr(license), str)


class TestCollections(PysticheTestCase):
    TEST_IMAGE_URL = "https://raw.githubusercontent.com/pmeier/pystiche/master/tests/assets/image/test_image.png"

    def test_DownloadableImage_generate_file(self):
        url = self.TEST_IMAGE_URL
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
            with self.subTest(title=title, author=author):
                actual = data.DownloadableImage.generate_file(
                    url, title=title, author=author
                )
                self.assertEqual(actual, desired)

    def test_DownloadableImage_download(self):
        with get_tmp_dir() as root:
            image = data.DownloadableImage(self.TEST_IMAGE_URL)
            image.download(root)

            file = path.join(root, image.file)
            self.assertTrue(path.exists(file))

            actual = self.load_image(file=file)
            desired = self.load_image()
            self.assertImagesAlmostEqual(actual, desired)

    def test_DownloadableImage_download_guides(self):
        with get_tmp_dir() as root:
            guide = data.DownloadableImage(
                self.TEST_IMAGE_URL,
                file="guide" + path.splitext(self.TEST_IMAGE_URL)[1],
            )
            image = data.DownloadableImage(
                self.TEST_IMAGE_URL,
                guides=data.DownloadableImageCollection({"guide": guide}),
            )
            image.download(root)

            actual = self.load_image(file=path.join(root, guide.file))
            desired = self.load_image()
            self.assertImagesAlmostEqual(actual, desired)

    def test_DownloadableImage_download_exist(self):
        with get_tmp_dir() as root:
            image = data.DownloadableImage(self.TEST_IMAGE_URL)
            file = path.join(root, image.file)

            shutil.copyfile(self.default_image_file(), file)

            with self.assertRaises(FileExistsError):
                image.download(root)

            image.md5 = "invalid_hash"
            with self.assertRaises(FileExistsError):
                image.download(root)

            image.md5 = "a858d33c424eaac1322cf3cab6d3d568"
            image.download(root)

    def test_DownloadableImage_download_overwrite(self):
        def create_fake_image(file):
            open(file, "wb").close()

        with get_tmp_dir() as root:
            image = data.DownloadableImage(self.TEST_IMAGE_URL)
            file = path.join(root, image.file)

            create_fake_image(file)
            image.download(root, overwrite=True)

            actual = self.load_image(file=file)
            desired = self.load_image()
            self.assertImagesAlmostEqual(actual, desired)

            create_fake_image(file)
            image.md5 = "a858d33c424eaac1322cf3cab6d3d568"
            image.download(root, overwrite=True)

            actual = self.load_image(file=file)
            desired = self.load_image()
            self.assertImagesAlmostEqual(actual, desired)

    def test_DownloadableImage_read(self):
        with get_tmp_dir() as root:
            image = data.DownloadableImage(self.TEST_IMAGE_URL)

            actual = image.read(root)
            desired = self.load_image()
            self.assertImagesAlmostEqual(actual, desired)

    def test_DownloadableImage_repr_smoke(self):
        image = data.DownloadableImage(self.TEST_IMAGE_URL)
        self.assertIsInstance(repr(image), str)

    def test_DownloadableImageCollection(self):
        with get_tmp_dir() as root:
            images = {"test_image": data.DownloadableImage(self.TEST_IMAGE_URL)}
            collection = data.DownloadableImageCollection(images,)
            collection.download(root=root)

            actual = collection["test_image"].read(root=root)
            desired = self.load_image()
            self.assertImagesAlmostEqual(actual, desired)

    def test_DownloadableImageCollection_read(self):
        with get_tmp_dir() as root:
            names = [str(idx) for idx in range(3)]
            collection = data.DownloadableImageCollection(
                {name: data.DownloadableImage(self.TEST_IMAGE_URL) for name in names}
            )
            images = collection.read(root)

            actual = images.keys()
            desired = names
            self.assertCountEqual(actual, desired)

            actuals = images.values()
            desired = self.load_image()
            for actual in actuals:
                self.assertImagesAlmostEqual(actual, desired)

    def test_LocalImage_collect_guides(self):
        def create_guides(root, file):
            dir = path.join(root, path.splitext(file)[0])
            os.makedirs(dir)
            src = self.default_image_file()
            files = {}
            for idx in range(3):
                region = str(idx)
                dst = path.join(dir, region + path.splitext(src)[1])
                copyfile(src, dst)
                files[region] = dst

            return files

        with get_tmp_dir() as root:
            file = "image.jpg"
            files = create_guides(root, file)

            image = data.LocalImage(path.join(root, file), collect_local_guides=True)

            self.assertIsNotNone(image.guides)

            actual = {region: guide.file for region, guide in image.guides}
            desired = files
            self.assertDictEqual(actual, desired)

    def test_LocalImage_collect_guides_no_dir(self):
        with get_tmp_dir() as root:
            image = data.LocalImage(
                path.join(root, "image.jpg"), collect_local_guides=True
            )

            self.assertIsNone(image.guides)

    def test_LocalImage_collect_guides_empty_dir(self):
        with get_tmp_dir() as root:
            file = "image.jpg"
            dir = path.join(root, path.splitext(file)[0])
            os.mkdir(dir)
            open(path.join(dir, "not_an_image.txt"), "wb").close()

            image = data.LocalImage(path.join(root, file), collect_local_guides=True)

            self.assertIsNone(image.guides)

    def test_LocalImage_read_abs_path(self):
        file = self.default_image_file()
        image = data.LocalImage(file)

        actual = image.read()
        desired = self.load_image()
        self.assertImagesAlmostEqual(actual, desired)

        actual = image.read(root="/invalid/root")
        desired = self.load_image()
        self.assertImagesAlmostEqual(actual, desired)

    def test_LocalImage_read_rel_path(self):
        file = self.default_image_file()
        root, filename = path.dirname(file), path.basename(file)
        image = data.LocalImage(filename)

        actual = image.read(root)
        desired = self.load_image()
        self.assertImagesAlmostEqual(actual, desired)

    def test_LocalImage_repr_smoke(self):
        image = data.LocalImage("image", transform=nn.Module(), note="note")
        self.assertIsInstance(repr(image), str)

    def test_LocalImageCollection_len(self):
        num_images = 3
        images = {str(idx): data.LocalImage(str(idx)) for idx in range(num_images)}
        collection = data.LocalImageCollection(images)

        actual = len(collection)
        desired = num_images
        self.assertEqual(actual, desired)

    def test_LocalImageCollection_getitem(self):
        images = {str(idx): data.LocalImage(str(idx)) for idx in range(3)}
        collection = data.LocalImageCollection(images)

        actual = {str(idx): collection[str(idx)] for idx in range(len(collection))}
        desired = images
        self.assertDictEqual(actual, desired)

    def test_LocalImageCollection_iter(self):
        images = {str(idx): data.LocalImage(str(idx)) for idx in range(3)}
        collection = data.LocalImageCollection(images)

        actual = {name: image for name, image in collection}
        desired = images
        self.assertDictEqual(actual, desired)

    def test_LocalImageCollection_read(self):
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

        with get_tmp_dir() as root:
            files = create_images(root)
            collection = data.LocalImageCollection(
                {name: data.LocalImage(file) for name, file in files.items()}
            )

            actual = collection.read()
            desired = {name: read_image(file) for name, file in files.items()}
            self.assertTensorDictAlmostEqual(actual, desired)

    def test_LocalImageCollection_repr_smoke(self):
        images = {str(idx): data.LocalImage(str(idx)) for idx in range(3)}
        collection = data.LocalImageCollection(images)

        self.assertIsInstance(repr(collection), str)
