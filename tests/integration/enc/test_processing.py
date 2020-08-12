from pystiche import enc
from pystiche.image import transforms


def test_get_preprocessor(subtests):
    frameworks_and_classes = (
        ("torch", transforms.TorchPreprocessing),
        ("caffe", transforms.CaffePreprocessing),
    )

    for framework, cls in frameworks_and_classes:
        assert isinstance(enc.get_preprocessor(framework), cls)
