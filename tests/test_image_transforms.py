import numpy as np
import pillow_affine as pa
from PIL import Image

import torch

from pystiche.image import (
    calculate_aspect_ratio,
    edge_to_image_size,
    extract_edge_size,
    extract_image_size,
    io,
    is_single_image,
    make_batched_image,
    make_single_image,
    transforms,
)

from .utils import PysticheTestCase


class PysticheTransfromTestCase(PysticheTestCase):
    def assertIdentityTransform(self, transform, image=None, tolerance=1e-2):
        if image is None:
            image = self.load_image()
        actual = image
        desired = transform(image)
        self.assertImagesAlmostEqual(actual, desired, tolerance=tolerance)

    @staticmethod
    def get_pil_affine_transform(affine_transform, expand=False):
        def transform(image):
            transform_params = affine_transform.extract_transform_params(
                image.size, expand=expand
            )
            return image.transform(*transform_params, resample=Image.BILINEAR)

        return transform

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


class TestColor(PysticheTransfromTestCase):
    def test_RGBToGrayscale(self):
        def PILRGBToGrayscale():
            def transform(image):
                assert image.mode == "RGB"
                return image.convert("L")

            return transform

        self.assertTransformEqualsPIL(
            pystiche_transform=transforms.RGBToGrayscale(),
            pil_transform=PILRGBToGrayscale(),
        )

    def test_GrayscaleToFakegrayscale(self):
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

    def test_RGBToFakegrayscale(self):
        def PILRGBToFakegrayscale():
            def transform(image):
                assert image.mode == "RGB"
                return image.convert("L").convert("RGB")

            return transform

        self.assertTransformEqualsPIL(
            pystiche_transform=transforms.RGBToFakegrayscale(),
            pil_transform=PILRGBToFakegrayscale(),
        )

    def test_GrayscaleToBinary(self):
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

    def test_RGBToBinary(self):
        def PILRGBToBinary():
            def transform(image):
                assert image.mode == "RGB"
                return image.convert("1", dither=0)

            return transform

        self.assertTransformEqualsPIL(
            pystiche_transform=transforms.RGBToBinary(), pil_transform=PILRGBToBinary()
        )

    def test_RGBToYUV(self):
        def PILRGBToYUV():
            def transform(image):
                assert image.mode == "RGB"
                # fmt: off
                matrix = (
                     0.299,  0.587,  0.114, 0.0,  # noqa: E126, E241
                    -0.147, -0.289,  0.436, 0.0,  # noqa: E131, E241
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

    def test_YUVToRGB(self):
        def transform(image):
            rgb_to_yuv = transforms.RGBToYUV()
            yuv_to_rgb = transforms.YUVToRGB()
            return yuv_to_rgb(rgb_to_yuv(image))

        self.assertIdentityTransform(transform, self.load_image())


class TestCore(PysticheTransfromTestCase):
    def test_Transform_add(self):
        class TestTransform(transforms.Transform):
            def forward(self):
                pass

        def add_transforms(transforms):
            added_transform = transforms[0]
            for transform in transforms[1:]:
                added_transform += transform
            return added_transform

        test_transforms = (TestTransform(), TestTransform(), TestTransform())
        added_transform = add_transforms(test_transforms)

        self.assertIsInstance(added_transform, transforms.ComposedTransform)
        for idx, test_transform in enumerate(test_transforms):
            actual = getattr(added_transform, str(idx))
            desired = test_transform
            self.assertIs(actual, desired)

    def test_ComposedTransform_call(self):
        class Plus(transforms.Transform):
            def __init__(self, plus):
                super().__init__()
                self.plus = plus

            def forward(self, input):
                return input + self.plus

        num_transforms = 3
        composed_transform = transforms.ComposedTransform(
            *[Plus(plus) for plus in range(1, num_transforms + 1)]
        )

        actual = composed_transform(0)
        desired = num_transforms * (num_transforms + 1) // 2
        self.assertEqual(actual, desired)

    def test_ComposedTransform_add(self):
        class TestTransform(transforms.Transform):
            def forward(self):
                pass

        test_transforms = (TestTransform(), TestTransform(), TestTransform())
        composed_transform = transforms.ComposedTransform(*test_transforms[:-1])
        single_transform = test_transforms[-1]
        added_transform = composed_transform + single_transform

        self.assertIsInstance(added_transform, transforms.ComposedTransform)
        for idx, test_transform in enumerate(test_transforms):
            actual = getattr(added_transform, str(idx))
            desired = test_transform
            self.assertIs(actual, desired)

    def test_compose_transforms_other(self):
        with self.assertRaises(TypeError):
            transforms.core.compose_transforms(None)


class TestCrop(PysticheTransfromTestCase):
    def test_Crop(self):
        image = self.load_image()
        origin = (200, 300)
        size = (50, 30)

        spatial_slices = {
            ("top", "left"): [
                slice(origin[0], origin[0] + size[0]),
                slice(origin[1], origin[1] + size[1]),
            ],
            ("bottom", "left"): [
                slice(origin[0] - size[0], origin[0]),
                slice(origin[1], origin[1] + size[1]),
            ],
            ("top", "right"): [
                slice(origin[0], origin[0] + size[0]),
                slice(origin[1] - size[1], origin[1]),
            ],
            ("bottom", "right"): [
                slice(origin[0] - size[0], origin[0]),
                slice(origin[1] - size[1], origin[1]),
            ],
        }

        for (vert_anchor, horz_anchor), spatial_slice in spatial_slices.items():
            with self.subTest(vert_anchor=vert_anchor, horz_anchor=horz_anchor):
                transform = transforms.Crop(
                    origin, size, vert_anchor=vert_anchor, horz_anchor=horz_anchor
                )
                actual = transform(image)
                desired = image[(slice(None), slice(None), *spatial_slice)]
                self.assertImagesAlmostEqual(actual, desired)

    def test_TopLeftCrop(self):
        image = self.load_image()
        size = 200

        transform = transforms.TopLeftCrop(size)
        actual = transform(image)
        desired = image[:, :, :size, :size]
        self.assertImagesAlmostEqual(actual, desired)

    def test_BottomLeftCrop(self):
        image = self.load_image()
        size = 200

        transform = transforms.BottomLeftCrop(size)
        actual = transform(image)
        desired = image[:, :, -size:, :size]
        self.assertImagesAlmostEqual(actual, desired)

    def test_TopRightCrop(self):
        image = self.load_image()
        size = 200

        transform = transforms.TopRightCrop(size)
        actual = transform(image)
        desired = image[:, :, :size, -size:]
        self.assertImagesAlmostEqual(actual, desired)

    def test_BottomRightCrop(self):
        image = self.load_image()
        size = 200

        transform = transforms.BottomRightCrop(size)
        actual = transform(image)
        desired = image[:, :, -size:, -size:]
        self.assertImagesAlmostEqual(actual, desired)

    def test_CenterCrop(self):
        image = torch.rand(1, 1, 100, 100)
        size = 50

        transform = transforms.CenterCrop(size)
        actual = transform(image)
        desired = image[:, :, size // 2 : -size // 2, size // 2 : -size // 2]
        self.assertImagesAlmostEqual(actual, desired)

    def test_ValidRandomCrop(self):
        def randint(range):
            return torch.randint(range + 1, ()).item()

        image_size = (100, 100)
        crop_size = (10, 20)
        image = torch.rand(1, 1, *image_size)

        image_height, image_width = image_size
        crop_height, crop_width = crop_size
        torch.manual_seed(0)
        vert_origin = randint(image_height - crop_height)
        horz_origin = randint(image_width - crop_width)

        torch.manual_seed(0)
        transform = transforms.ValidRandomCrop(crop_size)

        actual = transform(image)
        desired = image[
            :,
            :,
            vert_origin : vert_origin + crop_height,
            horz_origin : horz_origin + crop_width,
        ]
        self.assertImagesAlmostEqual(actual, desired)

    def test_ValidRandomCrop_identity(self):
        image = self.load_image()

        size = extract_image_size(image)
        transform = transforms.ValidRandomCrop(size)
        actual = transform(image)
        desired = image
        self.assertImagesAlmostEqual(actual, desired)

    def test_ValidRandomCrop_too_large(self):
        image = self.load_image()

        size = extract_edge_size(image, edge="long") * 2
        transform = transforms.ValidRandomCrop(size)

        with self.assertRaises(RuntimeError):
            transform(image)


class TestIo(PysticheTransfromTestCase):
    def test_ImportFromPIL_ExportToPIL_identity(self):
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

    def test_ImportFromPIL_single_image(self):
        import_transform = transforms.ImportFromPIL(make_batched=False)

        actual = import_transform(self.load_image(backend="PIL"))
        desired = self.load_single_image()
        self.assertImagesAlmostEqual(actual, desired)

    def test_ExportToPIL_multi_image(self):
        batch_size = 2
        export_transform = transforms.ExportToPIL()

        batched_image = self.load_batched_image(batch_size)
        actuals = export_transform(batched_image)
        desired = self.load_image(backend="PIL")

        self.assertTrue(isinstance(actuals, tuple))
        self.assertTrue(len(actuals) == batch_size)

        for actual in actuals:
            self.assertImagesAlmostEqual(actual, desired)


class TestMisc(PysticheTransfromTestCase):
    def test_FloatToUint8Range(self):
        image = torch.tensor(1.0)
        transform = transforms.FloatToUint8Range()

        actual = transform(image)
        desired = image * 255.0
        self.assertTensorAlmostEqual(actual, desired)

    def test_Uint8ToFloatRange(self):
        image = torch.tensor(255.0)
        transform = transforms.Uint8ToFloatRange()

        actual = transform(image)
        desired = image / 255.0
        self.assertTensorAlmostEqual(actual, desired)

    def test_FloatToUint8Range_Uint8ToFloatRange_identity(self):
        float_to_uint8_range = transforms.FloatToUint8Range()
        uint8_to_float_range = transforms.Uint8ToFloatRange()

        self.assertIdentityTransform(
            lambda image: uint8_to_float_range(float_to_uint8_range(image))
        )
        self.assertIdentityTransform(
            lambda image: float_to_uint8_range(uint8_to_float_range(image))
        )

    def test_ReverseChannelOrder(self):
        image = self.load_image()
        transform = transforms.ReverseChannelOrder()

        actual = transform(image)
        desired = image.flip(1)
        self.assertTensorAlmostEqual(actual, desired)

    def test_ReverseChannelOrder_identity(self):
        transform = transforms.ReverseChannelOrder()

        self.assertIdentityTransform(lambda image: transform(transform(image)))

    def test_Normalize(self):
        mean = (0.0, -1.0, 2.0)
        std = (1e0, 1e-1, 1e1)
        transform = transforms.Normalize(mean, std)

        torch.manual_seed(0)
        normalized_image = torch.randn((1, 3, 256, 256))

        def to_tensor(seq):
            return torch.tensor(seq).view(1, -1, 1, 1)

        image = normalized_image * to_tensor(std) + to_tensor(mean)

        actual = transform(image)
        desired = normalized_image
        self.assertTensorAlmostEqual(actual, desired, atol=1e-6)

    def test_Denormalize(self):
        mean = (0.0, -1.0, 2.0)
        std = (1e0, 1e-1, 1e1)
        transform = transforms.Denormalize(mean, std)

        torch.manual_seed(0)
        normalized_image = torch.randn((1, 3, 256, 256))

        def to_tensor(seq):
            return torch.tensor(seq).view(1, -1, 1, 1)

        image = normalized_image * to_tensor(std) + to_tensor(mean)

        actual = transform(normalized_image)
        desired = image
        self.assertTensorAlmostEqual(actual, desired, atol=1e-6)

    def test_Normalize_Denormalize_identity(self):
        mean = (0.0, -1.0, 2.0)
        std = (1e0, 1e-1, 1e1)

        normalize = transforms.Normalize(mean, std)
        denormalize = transforms.Denormalize(mean, std)

        self.assertIdentityTransform(lambda image: denormalize(normalize(image)))
        self.assertIdentityTransform(lambda image: normalize(denormalize(image)))


class TestMotif(PysticheTransfromTestCase):
    def test_ShearMotif(self):
        def PILShearMotif(angle, clockwise=False, center=None):
            if center is not None:
                center = tuple(center[::-1])
            affine_transform = pa.Shear(angle, clockwise=clockwise, center=center)
            return self.get_pil_affine_transform(affine_transform)

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

    def test_RotateMotif(self):
        def PILRotateMotif(angle, clockwise=False, center=None):
            if center is not None:
                center = tuple(center[::-1])
            affine_transform = pa.Rotate(angle, clockwise=clockwise, center=center)
            return self.get_pil_affine_transform(affine_transform)

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

    def test_ScaleMotif(self):
        def PILScaleMotif(factor, center=None):
            if not isinstance(factor, float):
                factor = tuple(factor[::-1])
            if center is not None:
                center = tuple(center[::-1])
            affine_transform = pa.Scale(factor, center=center)
            return self.get_pil_affine_transform(affine_transform)

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

    def test_TranslateMotif(self):
        def PILTranslateMotif(translation, inverse=False):
            translation = tuple(translation[::-1])
            affine_transform = pa.Translate(translation, inverse=inverse)
            return self.get_pil_affine_transform(affine_transform)

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

    def test_TransformMotifAffinely(self):
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
            return self.get_pil_affine_transform(affine_transform)

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

    def test_TransformMotifAffinely_empty(self):
        with self.assertRaises(RuntimeError):
            transforms.TransformMotifAffinely()

    def test_TransformMotifAffinely_full_canvas(self):
        def PILRotateMotif(angle, expand=False):
            affine_transform = pa.Rotate(angle)
            return self.get_pil_affine_transform(affine_transform, expand=expand)

        angle = 30.0
        pystiche_transform = transforms.RotateMotif(angle=angle, canvas="full")
        pil_transform = PILRotateMotif(angle=angle, expand=True)

        self.assertTransformEqualsPIL(
            pystiche_transform=pystiche_transform, pil_transform=pil_transform,
        )

    def test_TransformMotifAffinely_valid_canvas(self):
        pystiche_image = torch.ones(1, 1, 100, 100)

        angle = 45.0
        canvas = "valid"
        pystiche_transform = transforms.RotateMotif(angle=angle, canvas=canvas)
        with self.assertRaises(RuntimeError):
            pystiche_transform(pystiche_image)


class TestProcessing(PysticheTransfromTestCase):
    def test_TorchPreprocessing_TorchPostprocessing_identity(self):
        preprocessing_transform = transforms.TorchPreprocessing()
        postprocessing_transform = transforms.TorchPostprocessing()

        image = self.load_image()

        def pre_post_processing_transform(image):
            return postprocessing_transform(preprocessing_transform(image))

        self.assertIdentityTransform(pre_post_processing_transform, image)

        def post_pre_processing_transform(image):
            return preprocessing_transform(postprocessing_transform(image))

        self.assertIdentityTransform(post_pre_processing_transform, image)

        @transforms.torch_processing
        def identity(x):
            return x

        self.assertIdentityTransform(identity, image)

    def test_CaffePreprocessing_CaffePostprocessing_identity(self):
        preprocessing_transform = transforms.CaffePreprocessing()
        postprocessing_transform = transforms.CaffePostprocessing()

        image = self.load_image()

        def pre_post_processing_transform(image):
            return postprocessing_transform(preprocessing_transform(image))

        self.assertIdentityTransform(pre_post_processing_transform, image)

        def post_pre_processing_transform(image):
            return preprocessing_transform(postprocessing_transform(image))

        self.assertIdentityTransform(post_pre_processing_transform, image)

        @transforms.caffe_processing
        def identity(x):
            return x

        self.assertIdentityTransform(identity, image)


class TestResize(PysticheTransfromTestCase):
    def test_Resize_image_size(self):
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

    def test_Resize_edge_size(self):
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

    def test_Rescale(self):
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
