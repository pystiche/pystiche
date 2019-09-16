from os import path
import unittest
from PIL import Image
import numpy as np
import torch
from pystiche.image import calculate_aspect_ratio, edge_to_image_size, transforms


class Tester(unittest.TestCase):
    # The test image was downloaded from
    # http://www.r0k.us/graphics/kodak/kodim15.html
    # and is cleared for unrestricted usage
    TEST_IMAGE_FILE = path.join(path.dirname(__file__), "test_image.png")

    def get_pil_test_image(self):
        return Image.open(self.TEST_IMAGE_FILE)

    def get_pynst_test_image(self):
        processor = transforms.ImportFromPIL()
        return processor(self.get_pil_test_image())

    def assertImagesAlmostEqual(self, image1, image2, mean_tolerance=1e-2):
        def cast(image):
            if isinstance(image, Image.Image):
                mode = image.mode
                image = np.asarray(image, dtype=np.float32) / 255.0
                if mode == "L":
                    image = np.expand_dims(image, 2)
                return np.transpose(image, (2, 0, 1))
            elif isinstance(image, torch.Tensor):
                return image.squeeze(0).numpy()
            else:
                raise TypeError

        actual = np.mean(np.abs(cast(image1) - cast(image2)))
        desired = 0.0
        np.testing.assert_allclose(actual, desired, atol=mean_tolerance, rtol=0.0)

    def test_pil_import_export(self):
        pil_image = self.get_pil_test_image()

        import_transform = transforms.ImportFromPIL()
        export_transform = transforms.ExportToPIL()

        actual = export_transform(import_transform(pil_image))
        desired = pil_image
        self.assertImagesAlmostEqual(actual, desired)

    def test_resize(self):
        pystiche_image = self.get_pynst_test_image()
        pil_image = self.get_pil_test_image()

        def PILResizeTransform(image_size):
            size = image_size[::-1]
            return lambda image: image.resize(size, resample=Image.BILINEAR)

        image_size = (100, 100)
        pystiche_transform = transforms.Resize(image_size)
        pil_transform = PILResizeTransform(image_size)
        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(actual, desired, mean_tolerance=3e-2)

    def test_fixed_aspect_ratio_resize(self):
        pystiche_image = self.get_pynst_test_image()
        pil_image = self.get_pil_test_image()

        def PILFixedAspectRatioResizeTransform(edge_size, edge):
            def transform(image):
                aspect_ratio = calculate_aspect_ratio(image.size[::-1])
                image_size = edge_to_image_size(edge_size, aspect_ratio, edge)
                size = image_size[::-1]
                return image.resize(size, resample=Image.BILINEAR)

            return transform

        edge_size = 100
        for edge in ("short", "long", "vert", "horz"):
            pystiche_transform = transforms.FixedAspectRatioResize(edge_size, edge=edge)
            pil_transform = PILFixedAspectRatioResizeTransform(edge_size, edge=edge)
            actual = pystiche_transform(pystiche_image)
            desired = pil_transform(pil_image)
            self.assertImagesAlmostEqual(actual, desired, mean_tolerance=3e-2)

    def test_rescale(self):
        pystiche_image = self.get_pynst_test_image()
        pil_image = self.get_pil_test_image()

        def PILRescaleTransform(factor):
            def transform(image):
                size = [round(edge_size * factor) for edge_size in image.size]
                return image.resize(size, resample=Image.BILINEAR)

            return transform

        factor = 1.0 / np.pi
        pystiche_transform = transforms.Rescale(factor)
        pil_transform = PILRescaleTransform(factor)
        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(actual, desired, mean_tolerance=2e-2)

    def test_translate_motif(self):
        pystiche_image = self.get_pynst_test_image()
        pil_image = self.get_pil_test_image()

        def PILTranslateMotif(translation, inverse=False):
            if inverse:
                translation = [-val for val in translation]
            translate = (translation[0], -translation[1])
            return lambda image: image.rotate(
                0.0, translate=translate, resample=Image.BILINEAR
            )

        translation = (100.0, 100.0)
        pystiche_transform = transforms.TranslateMotif(translation)
        pil_transform = PILTranslateMotif(translation)
        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(actual, desired)

        inverse = True
        pystiche_transform = transforms.TranslateMotif(translation, inverse=inverse)
        pil_transform = PILTranslateMotif(translation, inverse=inverse)
        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(actual, desired)

    def test_rotate_motif(self):
        pystiche_image = self.get_pynst_test_image()
        pil_image = self.get_pil_test_image()

        def PILRotateMotif(angle, clockwise=False, center=None):
            if clockwise:
                angle *= -1.0
            if center is not None:
                center = (center[0], pil_image.height - center[1])
            return lambda image: image.rotate(
                angle, center=center, resample=Image.BILINEAR
            )

        angle = 30
        pystiche_transform = transforms.RotateMotif(angle)
        pil_transform = PILRotateMotif(angle)
        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(actual, desired)

        clockwise = True
        pystiche_transform = transforms.RotateMotif(angle, clockwise=clockwise)
        pil_transform = PILRotateMotif(angle, clockwise=clockwise)
        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(actual, desired)

        center = (0, 0)
        pystiche_transform = transforms.RotateMotif(angle, center=center)
        pil_transform = PILRotateMotif(angle, center=center)
        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(actual, desired)

    def test_transform_motif_affinely_crop(self):
        pystiche_image = torch.ones(1, 1, 100, 100)
        pil_image = transforms.ExportToPIL()(pystiche_image)

        def PILRotateMotif(angle, canvas):
            if canvas == "same":
                expand = False
            elif canvas == "full":
                expand = True
            else:
                raise ValueError
            return lambda image: image.rotate(
                angle, expand=expand, resample=Image.BILINEAR
            )

        # The PIL transform calculates the output image size differently than pystiche
        # so an off-by-one error might occur for different angles
        angle = 45.0
        canvas = "full"
        pystiche_transform = transforms.RotateMotif(angle=angle, canvas=canvas)
        pil_transform = PILRotateMotif(angle=angle, canvas=canvas)
        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(actual, desired)

    def test_transform_motif_affinely(self):
        pynst_image = torch.ones(1, 1, 100, 100)

        angle = 45.0
        canvas = "valid"
        pynst_transform = transforms.RotateMotif(angle=angle, canvas=canvas)
        self.assertRaises(RuntimeError, pynst_transform, pynst_image)


if __name__ == "__main__":
    unittest.main()
