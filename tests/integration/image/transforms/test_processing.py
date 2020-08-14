from pystiche.image import transforms

from . import assert_is_identity_transform


def test_TorchPreprocessing_TorchPostprocessing_identity(test_image):
    preprocessing_transform = transforms.TorchPreprocessing()
    postprocessing_transform = transforms.TorchPostprocessing()

    def pre_post_processing_transform(image):
        return postprocessing_transform(preprocessing_transform(image))

    assert_is_identity_transform(pre_post_processing_transform, test_image)

    def post_pre_processing_transform(image):
        return preprocessing_transform(postprocessing_transform(image))

    assert_is_identity_transform(post_pre_processing_transform, test_image)

    @transforms.torch_processing
    def identity(x):
        return x

    assert_is_identity_transform(identity, test_image)


def test_CaffePreprocessing_CaffePostprocessing_identity(test_image):
    preprocessing_transform = transforms.CaffePreprocessing()
    postprocessing_transform = transforms.CaffePostprocessing()

    def pre_post_processing_transform(image):
        return postprocessing_transform(preprocessing_transform(image))

    assert_is_identity_transform(pre_post_processing_transform, test_image)

    def post_pre_processing_transform(image):
        return preprocessing_transform(postprocessing_transform(image))

    assert_is_identity_transform(post_pre_processing_transform, test_image)

    @transforms.caffe_processing
    def identity(x):
        return x

    assert_is_identity_transform(identity, test_image)
