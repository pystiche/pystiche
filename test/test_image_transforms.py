import unittest
from PIL import Image
import numpy as np
import torch
from pystiche.image import (
    is_single_image,
    calculate_aspect_ratio,
    edge_to_image_size,
    make_single_image,
    make_batched_image,
    transforms,
    processing,
)
from pystiche.image.transforms import functional as F
from image_testcase import PysticheImageTestcase


class Tester(PysticheImageTestcase, unittest.TestCase):
    def assertTransformEqualsPIL(
        self,
        pystiche_transform,
        pil_transform,
        pystiche_image=None,
        pil_image=None,
        mean_abs_tolerance=1e-2,
    ):
        def parse_images(pystiche_image, pil_image):
            if pystiche_image is None and pil_image is None:
                pystiche_image = self.load_image("pystiche")
                pil_image = self.load_image("PIL")
            elif pystiche_image is None:
                pystiche_image = F.import_from_pil(pil_image)
            elif pil_image is None:
                pil_image = F.export_to_pil(pystiche_image)

            return pystiche_image, pil_image

        def get_single_and_batched_pystiche_images(pystiche_image):
            if is_single_image(pystiche_image):
                pystiche_single_image = pystiche_image
                pystiche_batched_image = make_batched_image(pystiche_single_image)
            else:
                pystiche_batched_image = pystiche_image
                pystiche_single_image = make_single_image(pystiche_batched_image)

            return pystiche_single_image, pystiche_batched_image

        def assert_transform_equality(pystiche_image, pil_image):
            actual = pystiche_transform(pystiche_image)
            desired = pil_transform(pil_image)
            self.assertImagesAlmostEqual(
                actual, desired, mean_abs_tolerance=mean_abs_tolerance
            )

        pystiche_image, pil_image = parse_images(pystiche_image, pil_image)

        (
            pystiche_single_image,
            pystiche_batched_image,
        ) = get_single_and_batched_pystiche_images(pystiche_image)

        assert_transform_equality(pystiche_single_image, pil_image)
        assert_transform_equality(pystiche_batched_image, pil_image)

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
        desired = self.load_single_image()
        self.assertImagesAlmostEqual(actual, desired)

    def test_multi_image_pil_export(self):
        batch_size = 2
        export_transform = transforms.ExportToPIL()

        batched_image = self.load_batched_image(batch_size)
        actuals = export_transform(batched_image)
        desired = self.load_image("PIL")

        self.assertTrue(isinstance(actuals, tuple))
        self.assertTrue(len(actuals) == batch_size)

        for actual in actuals:
            self.assertImagesAlmostEqual(actual, desired)

    def test_resize_with_image_size(self):
        def PILResizeTransform(image_size):
            size = image_size[::-1]
            return lambda image: image.resize(size, resample=Image.BILINEAR)

        image_size = (100, 100)
        pystiche_transform = transforms.Resize(image_size)
        pil_transform = PILResizeTransform(image_size)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform,
            pil_transform=pil_transform,
            mean_abs_tolerance=3e-2,
        )

    def test_resize_with_edge_size(self):
        def PILFixedAspectRatioResizeTransform(edge_size, edge):
            def transform(image):
                aspect_ratio = calculate_aspect_ratio(image.size[::-1])
                image_size = edge_to_image_size(edge_size, aspect_ratio, edge)
                size = image_size[::-1]
                return image.resize(size, resample=Image.BILINEAR)

            return transform

        edge_size = 100
        for edge in ("short", "long", "vert", "horz"):
            pystiche_transform = transforms.Resize(edge_size, edge=edge)
            pil_transform = PILFixedAspectRatioResizeTransform(edge_size, edge=edge)
            self.assertTransformEqualsPIL(
                pystiche_transform=pystiche_transform,
                pil_transform=pil_transform,
                mean_abs_tolerance=3e-2,
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
            mean_abs_tolerance=2e-2,
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
        pil_image = self.load_image("PIL")

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
                     0.615, -0.515, -0.100, 0.0,
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

    def test_torch_processing(self):
        preprocessing_transform = processing.TorchPreprocessing()
        postprocessing_transform = processing.TorchPostprocessing()

        image = self.load_image("pystiche")

        def pre_post_processing_transform(image):
            return postprocessing_transform(preprocessing_transform(image))

        self.assertIdentityTransform(pre_post_processing_transform, image)

        def post_pre_processing_transform(image):
            return preprocessing_transform(postprocessing_transform(image))

        self.assertIdentityTransform(post_pre_processing_transform, image)

        @processing.torch_processing
        def identity(x):
            return x

        self.assertIdentityTransform(identity, image)

    def test_caffe_processing(self):
        preprocessing_transform = processing.CaffePreprocessing()
        postprocessing_transform = processing.CaffePostprocessing()

        image = self.load_image("pystiche")

        def pre_post_processing_transform(image):
            return postprocessing_transform(preprocessing_transform(image))

        self.assertIdentityTransform(pre_post_processing_transform, image)

        def post_pre_processing_transform(image):
            return preprocessing_transform(postprocessing_transform(image))

        self.assertIdentityTransform(post_pre_processing_transform, image)

        @processing.caffe_processing
        def identity(x):
            return x

        self.assertIdentityTransform(identity, image)


if __name__ == "__main__":
    unittest.main()
