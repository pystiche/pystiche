from os import path
import unittest
from PIL import Image
import numpy as np
import torch
from pystiche.image import calculate_aspect_ratio, edge_to_image_size, transforms
from pystiche.image.transforms import functional as F


class Tester(unittest.TestCase):
    # The test image was downloaded from
    # http://www.r0k.us/graphics/kodak/kodim15.html
    # and is cleared for unrestricted usage
    TEST_IMAGE_FILE = path.join(path.dirname(__file__), "test_image.png")

    def pil_to_pystiche(self, image, device=None):
        if device is None:
            device = torch.device("cpu")
        return F.import_from_pil(image, device)

    def pystiche_to_pil(self, image, mode=None):
        return F.export_to_pil(image, mode=mode)

    def get_pil_test_image(self):
        return Image.open(self.TEST_IMAGE_FILE)

    def get_pystiche_test_image(self):
        return self.pil_to_pystiche(self.get_pil_test_image())

    def assertImagesAlmostEqual(self, image1, image2, mean_tolerance=1e-2):
        def cast(image):
            if isinstance(image, Image.Image):
                mode = image.mode
                image = np.asarray(image, dtype=np.float32)
                if mode in ("L", "RGB"):
                    image /= 255.0
                if mode in ("1", "L"):
                    image = np.expand_dims(image, 2)
                return np.transpose(image, (2, 0, 1))
            elif isinstance(image, torch.Tensor):
                return image.squeeze(0).numpy()
            else:
                raise TypeError

        actual = np.mean(np.abs(cast(image1) - cast(image2)))
        desired = 0.0
        np.testing.assert_allclose(actual, desired, atol=mean_tolerance, rtol=0.0)

    def assertTransformEqualsPIL(
        self,
        pystiche_transform,
        pil_transform,
        pystiche_image=None,
        pil_image=None,
        mean_tolerance=1e-2,
    ):
        if pil_image is None:
            pil_image = self.get_pil_test_image()
        if pystiche_image is None:
            pystiche_image = self.pil_to_pystiche(pil_image)

        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(actual, desired, mean_tolerance=mean_tolerance)

    def assertIdentityTransform(self, transform, image, mean_tolerance=1e-2):
        actual = image
        desired = transform(image)
        self.assertImagesAlmostEqual(actual, desired, mean_tolerance=mean_tolerance)

    def test_pil_import_export(self):
        import_transform = transforms.ImportFromPIL()
        export_transform = transforms.ExportToPIL()

        def import_export_transform(image):
            return export_transform(import_transform(image))

        self.assertIdentityTransform(import_export_transform, self.get_pil_test_image())

        def export_import_transform(image):
            return import_transform(export_transform(image))

        self.assertIdentityTransform(
            export_import_transform, self.get_pystiche_test_image()
        )

    def test_resize(self):
        def PILResizeTransform(image_size):
            size = image_size[::-1]
            return lambda image: image.resize(size, resample=Image.BILINEAR)

        image_size = (100, 100)
        pystiche_transform = transforms.Resize(image_size)
        pil_transform = PILResizeTransform(image_size)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform,
            pil_transform=pil_transform,
            mean_tolerance=3e-2,
        )

    def test_fixed_aspect_ratio_resize(self):
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
            self.assertTransformEqualsPIL(
                pystiche_transform=pystiche_transform,
                pil_transform=pil_transform,
                mean_tolerance=3e-2,
            )

    def test_rescale(self):
        def PILRescaleTransform(factor):
            def transform(image):
                size = [round(edge_size * factor) for edge_size in image.size]
                return image.resize(size, resample=Image.BILINEAR)

            return transform

        factor = 1.0 / np.pi
        pystiche_transform = transforms.Rescale(factor)
        pil_transform = PILRescaleTransform(factor)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform,
            pil_transform=pil_transform,
            mean_tolerance=2e-2,
        )

    def test_translate_motif(self):
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
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform
        )

        inverse = True
        pystiche_transform = transforms.TranslateMotif(translation, inverse=inverse)
        pil_transform = PILTranslateMotif(translation, inverse=inverse)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform
        )

    def test_rotate_motif(self):
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
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform,
            pil_transform=pil_transform,
            pil_image=pil_image,
        )

        clockwise = True
        pystiche_transform = transforms.RotateMotif(angle, clockwise=clockwise)
        pil_transform = PILRotateMotif(angle, clockwise=clockwise)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform,
            pil_transform=pil_transform,
            pil_image=pil_image,
        )

        center = (0, 0)
        pystiche_transform = transforms.RotateMotif(angle, center=center)
        pil_transform = PILRotateMotif(angle, center=center)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform,
            pil_transform=pil_transform,
            pil_image=pil_image,
        )

    # FIXME: rename
    def test_transform_motif_affinely_crop(self):
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

        pystiche_image = torch.ones(1, 1, 100, 100)
        pil_image = transforms.ExportToPIL()(pystiche_image)

        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform,
            pil_transform=pil_transform,
            pystiche_image=pystiche_image,
            pil_image=pil_image,
        )

        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(actual, desired)

    # FIXME: rename
    def test_transform_motif_affinely(self):
        pynst_image = torch.ones(1, 1, 100, 100)

        angle = 45.0
        canvas = "valid"
        pynst_transform = transforms.RotateMotif(angle=angle, canvas=canvas)
        self.assertRaises(RuntimeError, pynst_transform, pynst_image)

    def test_rgb_to_grayscale(self):
        def PILRGBToGrayscale():
            def transform(image):
                assert image.mode == "RGB"
                return image.convert("L")

            return transform

        self.assertTransformEqualsPIL(
            pystiche_transform=transforms.RGBToGrayscale(),
            pil_transform=PILRGBToGrayscale(),
        )

    def test_grayscale_to_fakegrayscale(self):
        def PILGrayscaleToFakegrayscale():
            def transform(image):
                assert image.mode == "L"
                return image.convert("RGB")

            return transform

        self.assertTransformEqualsPIL(
            pystiche_transform=transforms.GrayscaleToFakegrayscale(),
            pil_transform=PILGrayscaleToFakegrayscale(),
            pil_image=self.get_pil_test_image().convert("L"),
        )

    def test_rgb_to_fakegrayscale(self):
        def PILRGBToFakegrayscale():
            def transform(image):
                assert image.mode == "RGB"
                return image.convert("L").convert("RGB")

            return transform

        self.assertTransformEqualsPIL(
            pystiche_transform=transforms.RGBToFakegrayscale(),
            pil_transform=PILRGBToFakegrayscale(),
        )

    def test_grayscale_to_binary(self):
        def PILGrayscaleToBinary():
            def transform(image):
                assert image.mode == "L"
                return image.convert("1", dither=0)

            return transform

        self.assertTransformEqualsPIL(
            pystiche_transform=transforms.GrayscaleToBinary(),
            pil_transform=PILGrayscaleToBinary(),
            pil_image=self.get_pil_test_image().convert("L"),
        )

    def test_rgb_to_binary(self):
        def PILRGBToBinary():
            def transform(image):
                assert image.mode == "RGB"
                return image.convert("1", dither=0)

            return transform

        self.assertTransformEqualsPIL(
            pystiche_transform=transforms.RGBToBinary(), pil_transform=PILRGBToBinary()
        )

    def test_rgb_to_yuv(self):
        def PILRGBToYUV():
            def transform(image):
                assert image.mode == "RGB"
                # fmt: off
                matrix = (
                     0.299,  0.587,  0.114, 0.0,
                    -0.147, -0.289,  0.436, 0.0,
                     0.615, -0.515, -0.100, 0.0
                )
                # fmt: on
                return image.convert("RGB", matrix)

            return transform

        self.assertTransformEqualsPIL(
            pystiche_transform=transforms.RGBToYUV(),
            pil_transform=PILRGBToYUV(),
            mean_tolerance=2e-2,
        )

    def test_yuv_to_rgb(self):
        def transform(image):
            rgb_to_yuv = transforms.RGBToYUV()
            yuv_to_rgb = transforms.YUVToRGB()
            return yuv_to_rgb(rgb_to_yuv(image))

        self.assertIdentityTransform(transform, self.get_pystiche_test_image())


if __name__ == "__main__":
    unittest.main()
