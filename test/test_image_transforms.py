import unittest
from PIL import Image
import numpy as np
import torch
from pystiche.image import calculate_aspect_ratio, edge_to_image_size, transforms
from pystiche.image.transforms import functional as F
from utils import PysticheImageTestcase


class Tester(PysticheImageTestcase, unittest.TestCase):
    def assertTransformEqualsPIL(
        self,
        pystiche_transform,
        pil_transform,
        pystiche_image=None,
        pil_image=None,
        mean_abs_tolerance=1e-2,
    ):
        if pil_image is None and pystiche_image is None:
            pil_image = self.load_image("PIL")
            pystiche_image = self.load_image("pystiche")
        elif pil_image is None:
            pil_image = F.export_to_pil(pystiche_image)
        elif pystiche_image is None:
            pystiche_image = F.import_from_pil(pil_image, torch.device("cpu"))

        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        self.assertImagesAlmostEqual(
            actual, desired, mean_abs_tolerance=mean_abs_tolerance
        )

    def assertIdentityTransform(self, transform, image, mean_abs_tolerance=1e-2):
        actual = image
        desired = transform(image)
        self.assertImagesAlmostEqual(
            actual, desired, mean_abs_tolerance=mean_abs_tolerance
        )

    def test_pil_import_export(self):
        import_transform = transforms.ImportFromPIL()
        export_transform = transforms.ExportToPIL()

        def import_export_transform(image):
            return export_transform(import_transform(image))

        self.assertIdentityTransform(import_export_transform, self.load_image("PIL"))

        def export_import_transform(image):
            return import_transform(export_transform(image))

        self.assertIdentityTransform(
            export_import_transform, self.load_image("pystiche")
        )

    def test_single_image_pil_import(self):
        import_transform = transforms.ImportFromPIL(make_batched=False)

        actual = import_transform(self.load_image("PIL"))
        desired = self.load_image("pystiche").squeeze(0)
        self.assertImagesAlmostEqual(actual, desired)

    def test_multi_image_pil_export(self):
        batch_size = 2
        export_transform = transforms.ExportToPIL()

        batched_image = self.load_image("pystiche").repeat(batch_size, 1, 1, 1)
        actuals = export_transform(batched_image)
        desired = self.load_image("PIL")

        self.assertTrue(isinstance(actuals, tuple))
        self.assertTrue(len(actuals) == batch_size)

        for actual in actuals:
            self.assertImagesAlmostEqual(actual, desired)

    def test_grayscale_to_fakegrayscale(self):
        def PILGrayscaleToFakegrayscale():
            def transform(image):
                assert image.mode == "L"
                return image.convert("RGB")

            return transform

        self.assertTransformEqualsPIL(
            pystiche_transform=transforms.GrayscaleToFakegrayscale(),
            pil_transform=PILGrayscaleToFakegrayscale(),
            pil_image=self.load_image("PIL").convert("L"),
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
            pil_image=self.load_image("PIL").convert("L"),
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
            mean_abs_tolerance=2e-2,
        )

    def test_yuv_to_rgb(self):
        def transform(image):
            rgb_to_yuv = transforms.RGBToYUV()
            yuv_to_rgb = transforms.YUVToRGB()
            return yuv_to_rgb(rgb_to_yuv(image))

        self.assertIdentityTransform(transform, self.load_image("pystiche"))


if __name__ == "__main__":
    unittest.main()
