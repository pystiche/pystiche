import contextlib
import os
from os import path

from pystiche import data
from utils import PysticheTestCase, get_tmp_dir


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


class TestSampler(PysticheTestCase):
    def test_InfiniteCycleBatchSampler(self):
        data_source = [None] * 3
        batch_size = 2

        batch_sampler = data.InfiniteCycleBatchSampler(
            data_source, batch_size=batch_size
        )

        actual = []
        for idx, batch in enumerate(batch_sampler):
            if idx == 6:
                break
            actual.append(batch)
        actual = tuple(actual)

        desired = ((0, 1), (2, 0), (1, 2)) * 2
        self.assertEqual(actual, desired)

    def test_FiniteCycleBatchSampler(self):
        data_source = [None] * 3
        num_batches = 6
        batch_size = 2

        batch_sampler = data.FiniteCycleBatchSampler(
            data_source, num_batches, batch_size=batch_size
        )

        actual = tuple(iter(batch_sampler))
        desired = ((0, 1), (2, 0), (1, 2)) * 2
        self.assertEqual(actual, desired)

    def test_InfiniteCycleBatchSampler_len(self):
        data_source = [None] * 3
        num_batches = 2
        batch_sampler = data.FiniteCycleBatchSampler(data_source, num_batches)
        self.assertEqual(len(batch_sampler), num_batches)
