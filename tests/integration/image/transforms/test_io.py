import pyimagetest

from pystiche import image
from pystiche.image import transforms

from . import assert_is_identity_transform


def test_ImportFromPIL_ExportToPIL_identity(test_image, test_image_pil):
    import_transform = transforms.ImportFromPIL()
    export_transform = transforms.ExportToPIL()

    def import_export_transform(image):
        return export_transform(import_transform(image))

    assert_is_identity_transform(import_export_transform, test_image_pil)

    def export_import_transform(image):
        return import_transform(export_transform(image))

    assert_is_identity_transform(export_import_transform, test_image)


def test_ImportFromPIL_single_image(test_image, test_image_pil):
    import_transform = transforms.ImportFromPIL(make_batched=False)

    actual = import_transform(test_image_pil)
    desired = image.make_single_image(test_image)
    pyimagetest.assert_images_almost_equal(actual, desired)


def test_ExportToPIL_multi_image(test_image, test_image_pil):
    batch_size = 2
    export_transform = transforms.ExportToPIL()

    batched_image = test_image.repeat(2, 1, 1, 1)
    actuals = export_transform(batched_image)
    desired = test_image_pil

    assert isinstance(actuals, tuple)
    assert len(actuals) == batch_size

    for actual in actuals:
        pyimagetest.assert_images_almost_equal(actual, desired)
