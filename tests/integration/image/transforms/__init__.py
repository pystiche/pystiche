import functools

import pyimagetest

from pystiche import image as image_

from tests.assets import read_image as _read_image

read_image = functools.partial(_read_image, "test_image")

__all__ = ["assert_is_identity_transform", "assert_transform_equals_pil"]


def assert_is_identity_transform(transform, image=None, mae=1e-2):
    if image is None:
        image = read_image()
    pyimagetest.assert_images_almost_equal(image, transform(image), mae=mae)


def assert_transform_equals_pil(
    pystiche_transform, pil_transform, pystiche_image=None, pil_image=None, mae=1e-2,
):
    def parse_images(pystiche_image, pil_image):
        if pystiche_image is None and pil_image is None:
            pystiche_image = read_image()
            pil_image = read_image(pil=True)
        elif pystiche_image is None:
            pystiche_image = image_.import_from_pil(pil_image)
        elif pil_image is None:
            pil_image = image_.export_to_pil(pystiche_image)

        return pystiche_image, pil_image

    def get_single_and_batched_pystiche_images(pystiche_image):
        if image_.is_single_image(pystiche_image):
            pystiche_single_image = pystiche_image
            pystiche_batched_image = image_.make_batched_image(pystiche_single_image)
        else:
            pystiche_batched_image = pystiche_image
            pystiche_single_image = image_.make_single_image(pystiche_batched_image)

        return pystiche_single_image, pystiche_batched_image

    def assert_transform_equality(pystiche_image, pil_image):
        actual = pystiche_transform(pystiche_image)
        desired = pil_transform(pil_image)
        pyimagetest.assert_images_almost_equal(actual, desired, mae=mae)

    pystiche_image, pil_image = parse_images(pystiche_image, pil_image)

    (
        pystiche_single_image,
        pystiche_batched_image,
    ) = get_single_and_batched_pystiche_images(pystiche_image)

    assert_transform_equality(pystiche_single_image, pil_image)
    assert_transform_equality(pystiche_batched_image, pil_image)
