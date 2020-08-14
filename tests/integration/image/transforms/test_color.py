from pystiche.image import transforms

from . import assert_is_identity_transform, assert_transform_equals_pil


def test_RGBToGrayscale():
    def PILRGBToGrayscale():
        def transform(image):
            assert image.mode == "RGB"
            return image.convert("L")

        return transform

    assert_transform_equals_pil(
        pystiche_transform=transforms.RGBToGrayscale(),
        pil_transform=PILRGBToGrayscale(),
    )


def test_GrayscaleToFakegrayscale(test_image_pil):
    def PILGrayscaleToFakegrayscale():
        def transform(image):
            assert image.mode == "L"
            return image.convert("RGB")

        return transform

    assert_transform_equals_pil(
        pystiche_transform=transforms.GrayscaleToFakegrayscale(),
        pil_transform=PILGrayscaleToFakegrayscale(),
        pil_image=test_image_pil.convert("L"),
    )


def test_RGBToFakegrayscale():
    def PILRGBToFakegrayscale():
        def transform(image):
            assert image.mode == "RGB"
            return image.convert("L").convert("RGB")

        return transform

    assert_transform_equals_pil(
        pystiche_transform=transforms.RGBToFakegrayscale(),
        pil_transform=PILRGBToFakegrayscale(),
    )


def test_GrayscaleToBinary(test_image_pil):
    def PILGrayscaleToBinary():
        def transform(image):
            assert image.mode == "L"
            return image.convert("1", dither=0)

        return transform

    assert_transform_equals_pil(
        pystiche_transform=transforms.GrayscaleToBinary(),
        pil_transform=PILGrayscaleToBinary(),
        pil_image=test_image_pil.convert("L"),
    )


def test_RGBToBinary():
    def PILRGBToBinary():
        def transform(image):
            assert image.mode == "RGB"
            return image.convert("1", dither=0)

        return transform

    assert_transform_equals_pil(
        pystiche_transform=transforms.RGBToBinary(), pil_transform=PILRGBToBinary()
    )


def test_RGBToYUV():
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

    assert_transform_equals_pil(
        pystiche_transform=transforms.RGBToYUV(), pil_transform=PILRGBToYUV(), mae=2e-2,
    )


def test_YUVToRGB():
    def transform(image):
        rgb_to_yuv = transforms.RGBToYUV()
        yuv_to_rgb = transforms.YUVToRGB()
        return yuv_to_rgb(rgb_to_yuv(image))

    assert_is_identity_transform(transform)
