import pytest

from torch import hub

from pystiche.enc.models.vgg import MODEL_URLS


@pytest.mark.slow
def test_vgg(subtests, github):
    def should_be_available(arch, framework):
        if arch in ("vgg16", "vgg19") and framework == "caffe":
            return True

        return framework == "torch"

    for arch, framework in MODEL_URLS.keys():
        with subtests.test(arch=arch, framework=framework):
            load = lambda: hub.load(  # noqa: E731
                github, arch, pretrained=True, framework=framework
            )
            if should_be_available(arch, framework):
                assert load()
            else:
                with pytest.raises(KeyError):
                    load()
