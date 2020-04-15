import itertools
import shutil
from os import path

from pystiche import data
from utils import PysticheTestCase, get_tmp_dir

TEST_IMAGE_URL = "https://raw.githubusercontent.com/pmeier/pystiche/master/tests/assets/image/test_image.png"


class TestCollection(PysticheTestCase):
    def test_ImageCollection_len(self):
        num_images = 3
        images = {str(idx): data.Image(str(idx)) for idx in range(num_images)}
        collection = data.ImageCollection(images)

        actual = len(collection)
        desired = num_images
        self.assertEqual(actual, desired)

    def test_ImageCollection_getitem(self):
        images = {str(idx): data.Image(str(idx)) for idx in range(3)}
        collection = data.ImageCollection(images)

        actual = {str(idx): collection[str(idx)] for idx in range(len(collection))}
        desired = images
        self.assertDictEqual(actual, desired)

    def test_ImageCollection_repr_smoke(self):
        images = {str(idx): data.Image(str(idx)) for idx in range(3)}
        collection = data.ImageCollection(images)

        self.assertIsInstance(repr(collection), str)

    def test_DownloadableImageCollection(self):
        with get_tmp_dir() as root:
            images = {"test_image": data.DownloadableImage(TEST_IMAGE_URL)}
            collection = data.DownloadableImageCollection(images, root=root)

            actual = collection["test_image"].read()
            desired = self.load_image()
            self.assertImagesAlmostEqual(actual, desired)


class TestImage(PysticheTestCase):
    def test_Image_read(self):
        file = self.default_image_file()
        root, filename = path.dirname(file), path.basename(file)
        image = data.Image(filename)

        actual = image.read(root)
        desired = self.load_image()
        self.assertImagesAlmostEqual(actual, desired)

    def test_Image_repr_smoke(self):
        image = data.Image("image")
        self.assertIsInstance(repr(image), str)

    def test_DownloadableImage_generate_file(self):
        url = TEST_IMAGE_URL
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
            image = data.DownloadableImage(TEST_IMAGE_URL)
            image.download(root)

            file = path.join(root, image.file)
            self.assertTrue(path.exists(file))

            actual = self.load_image(file=file)
            desired = self.load_image()
            self.assertImagesAlmostEqual(actual, desired)

    def test_DownloadableImage_download_exist(self):
        with get_tmp_dir() as root:
            image = data.DownloadableImage(TEST_IMAGE_URL)
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
            with open(file, "wb"):
                pass

        with get_tmp_dir() as root:
            image = data.DownloadableImage(TEST_IMAGE_URL)
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
            image = data.DownloadableImage(TEST_IMAGE_URL)

            actual = image.read(root)
            desired = self.load_image()
            self.assertImagesAlmostEqual(actual, desired)

    def test_DownloadableImage_repr_smoke(self):
        image = data.DownloadableImage(TEST_IMAGE_URL)
        self.assertIsInstance(repr(image), str)
