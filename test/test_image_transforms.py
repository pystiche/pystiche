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
    io,
    transforms,
    processing,
)
import pillow_affine as pa
from image_test_case import PysticheImageTestCase


def PILTransform(affine_transform, expand=False):
    def transform(image):
        transform_params = affine_transform.extract_transform_params(
            image.size, expand=expand
        )
        return image.transform(*transform_params, resample=Image.BILINEAR)

    return transform


class Tester(PysticheImageTestCase, unittest.TestCase):
    def assertTransformEqualsPIL(
        self,
        pystiche_transform,
        pil_transform,
        pystiche_image=None,
        pil_image=None,
        tolerance=1e-2,
    ):
        def parse_images(pystiche_image, pil_image):
            if pystiche_image is None and pil_image is None:
                pystiche_image = self.load_image(backend="pystiche")
                pil_image = self.load_image(backend="PIL")
            elif pystiche_image is None:
                pystiche_image = io.import_from_pil(pil_image)
            elif pil_image is None:
                pil_image = io.export_to_pil(pystiche_image)

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
            self.assertImagesAlmostEqual(actual, desired, tolerance=tolerance)

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

        self.assertIdentityTransform(
            import_export_transform, self.load_image(backend="PIL")
        )

        def export_import_transform(image):
            return import_transform(export_transform(image))

        self.assertIdentityTransform(
            export_import_transform, self.load_image(backend="pystiche")
        )

    def test_single_image_pil_import(self):
        import_transform = transforms.ImportFromPIL(make_batched=False)

        actual = import_transform(self.load_image(backend="PIL"))
        desired = self.load_single_image()
        self.assertImagesAlmostEqual(actual, desired)

    def test_multi_image_pil_export(self):
        batch_size = 2
        export_transform = transforms.ExportToPIL()

        batched_image = self.load_batched_image(batch_size)
        actuals = export_transform(batched_image)
        desired = self.load_image(backend="PIL")

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
            tolerance=3e-2,
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
                tolerance=3e-2,
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
            tolerance=2e-2,
        )

    def test_shear_motif(self):
        def PILShearMotif(angle, clockwise=False, center=None):
            if center is not None:
                center = tuple(center[::-1])
            affine_transform = pa.Shear(angle, clockwise=clockwise, center=center)
            return PILTransform(affine_transform)

        angle = 30
        pystiche_transform = transforms.ShearMotif(angle)
        pil_transform = PILShearMotif(angle)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

        clockwise = True
        pystiche_transform = transforms.ShearMotif(angle, clockwise=clockwise)
        pil_transform = PILShearMotif(angle, clockwise=clockwise)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

        center = (100.0, 50.0)
        pystiche_transform = transforms.ShearMotif(angle, center=center)
        pil_transform = PILShearMotif(angle, center=center)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

    def test_rotate_motif(self):
        def PILRotateMotif(angle, clockwise=False, center=None):
            if center is not None:
                center = tuple(center[::-1])
            affine_transform = pa.Rotate(angle, clockwise=clockwise, center=center)
            return PILTransform(affine_transform)

        angle = 30
        pystiche_transform = transforms.RotateMotif(angle)
        pil_transform = PILRotateMotif(angle)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

        clockwise = True
        pystiche_transform = transforms.RotateMotif(angle, clockwise=clockwise)
        pil_transform = PILRotateMotif(angle, clockwise=clockwise)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

        center = (100.0, 50.0)
        pystiche_transform = transforms.RotateMotif(angle, center=center)
        pil_transform = PILRotateMotif(angle, center=center)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

    def test_scale_motif(self):
        def PILScaleMotif(factor, center=None):
            if not isinstance(factor, float):
                factor = tuple(factor[::-1])
            if center is not None:
                center = tuple(center[::-1])
            affine_transform = pa.Scale(factor, center=center)
            return PILTransform(affine_transform)

        factor = 2.0
        pystiche_transform = transforms.ScaleMotif(factor)
        pil_transform = PILScaleMotif(factor)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

        factor = (0.3, 0.7)
        pystiche_transform = transforms.ScaleMotif(factor)
        pil_transform = PILScaleMotif(factor)
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

        factor = 0.5
        center = (100.0, 50.0)
        pystiche_transform = transforms.ScaleMotif(factor, center=center)
        pil_transform = PILScaleMotif(factor, center=center)
        pystiche_transform(torch.rand(1, 100, 100))
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

    def test_translate_motif(self):
        def PILTranslateMotif(translation, inverse=False):
            translation = tuple(translation[::-1])
            affine_transform = pa.Translate(translation, inverse=inverse)
            return PILTransform(affine_transform)

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

    def test_transform_motif_affinely_same_canvas(self):
        def PILTransformMotifAffinely(
            shearing_angle=0.0,
            clockwise_shearing=False,
            shearing_center=None,
            rotation_angle=0.0,
            clockwise_rotation=False,
            rotation_center=None,
            scaling_factor=1.0,
            scaling_center=None,
            translation=(0.0, 0.0),
            inverse_translation=False,
        ):
            if shearing_center is not None:
                shearing_center = tuple(shearing_center[::-1])
            if rotation_center is not None:
                rotation_center = tuple(rotation_center[::-1])
            if scaling_center is not None:
                scaling_center = tuple(scaling_center[::-1])
            if not isinstance(scaling_factor, float):
                scaling_factor = tuple(scaling_factor[::-1])
            translation = tuple(translation[::-1])
            affine_transform = pa.ComposedTransform(
                pa.Shear(
                    shearing_angle, clockwise=clockwise_shearing, center=shearing_center
                ),
                pa.Rotate(
                    rotation_angle, clockwise=clockwise_rotation, center=rotation_center
                ),
                pa.Scale(scaling_factor, center=scaling_center),
                pa.Translate(translation, inverse=inverse_translation),
            )
            return PILTransform(affine_transform)

        transform_kwargs = {
            "shearing_angle": 10.0,
            "clockwise_shearing": True,
            "shearing_center": None,
            "rotation_angle": 20.0,
            "clockwise_rotation": True,
            "rotation_center": (0.0, 0.0),
            "scaling_factor": (0.6, 0.7),
            "scaling_center": (300.0, 200.0),
            "translation": (-20.0, 50.0),
            "inverse_translation": False,
        }

        pystiche_transform = transforms.TransformMotifAffinely(**transform_kwargs)
        pil_transform = PILTransformMotifAffinely(**transform_kwargs)
        pystiche_transform(torch.rand(1, 100, 100))
        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform
        )

    def test_transform_motif_affinely_full_canvas(self):
        def PILRotateMotif(angle, expand=False):
            affine_transform = pa.Rotate(angle)
            return PILTransform(affine_transform, expand=expand)

        angle = 30.0
        pystiche_transform = transforms.RotateMotif(angle=angle, canvas="full")
        pil_transform = PILRotateMotif(angle=angle, expand=True)

        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

    def test_transform_motif_affinely_valid_canvas(self):
        pystiche_image = torch.ones(1, 1, 100, 100)

        angle = 45.0
        canvas = "valid"
        pystiche_transform = transforms.RotateMotif(angle=angle, canvas=canvas)
        with self.assertRaises(RuntimeError):
            pystiche_transform(pystiche_image)

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
            pil_image=self.load_image(backend="PIL").convert("L"),
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
            pil_image=self.load_image(backend="PIL").convert("L"),
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
            tolerance=2e-2,
        )

    def test_yuv_to_rgb(self):
        def transform(image):
            rgb_to_yuv = transforms.RGBToYUV()
            yuv_to_rgb = transforms.YUVToRGB()
            return yuv_to_rgb(rgb_to_yuv(image))

        self.assertIdentityTransform(transform, self.load_image())

    def test_torch_processing(self):
        preprocessing_transform = processing.TorchPreprocessing()
        postprocessing_transform = processing.TorchPostprocessing()

        image = self.load_image()

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

        image = self.load_image()

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

    def test_top_left_crop(self):
        image = self.load_image()
        size = 200

        transform = transforms.TopLeftCrop(size)
        actual = transform(image)
        desired = image[:, :, :size, :size]
        self.assertImagesAlmostEqual(actual, desired)

    def test_bottom_left_crop(self):
        image = self.load_image()
        size = 200

        transform = transforms.BottomLeftCrop(size)
        actual = transform(image)
        desired = image[:, :, -size:, :size]
        self.assertImagesAlmostEqual(actual, desired)

    def test_top_right_crop(self):
        image = self.load_image()
        size = 200

        transform = transforms.TopRightCrop(size)
        actual = transform(image)
        desired = image[:, :, :size, -size:]
        self.assertImagesAlmostEqual(actual, desired)

    def test_bottom_right_crop(self):
        image = self.load_image()
        size = 200

        transform = transforms.BottomRightCrop(size)
        actual = transform(image)
        desired = image[:, :, -size:, -size:]
        self.assertImagesAlmostEqual(actual, desired)

    def test_center_crop(self):
        image = torch.rand(1, 1, 100, 100)
        size = 50

        transform = transforms.CenterCrop(size)
        actual = transform(image)
        desired = image[:, :, size // 2 : -size // 2, size // 2 : -size // 2]
        self.assertImagesAlmostEqual(actual, desired)


if __name__ == "__main__":
    unittest.main()
