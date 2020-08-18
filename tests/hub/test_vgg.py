import pytest

from torch import hub

from pystiche.enc.models.vgg import MODEL_URLS


@pytest.mark.slow
def test_vgg_caffe(subtests, github):
    for framework, arch in MODEL_URLS.keys():
        if framework != "caffe":
            continue

        with subtests.test(arch=arch):
            assert hub.load(github, f"{arch}_caffe")
